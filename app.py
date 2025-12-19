#!/usr/bin/env python3
"""
FACS Autogating - Gates Hexagonaux avec Édition sur Graphique
- Gates hexagonaux (6 sommets) bien positionnés
- Cliquer sur un sommet pour le sélectionner, cliquer ailleurs pour le déplacer
- Mise à jour en cascade de toutes les populations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from pathlib import Path
import tempfile
import io
import json
import os
import flowio

try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except ImportError:
    HAS_PLOTLY_EVENTS = False

st.set_page_config(page_title="FACS - Hexagones", page_icon="🔬", layout="wide")

LEARNED_PARAMS_FILE = "learned_gating_params.json"

st.markdown("""
<style>
.main-header { font-size: 1.8rem; color: #2c3e50; text-align: center; margin-bottom: 0.5rem; }
.info-box { background: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid #0066cc; margin: 0.5rem 0; }
.selected-box { background: #d4edda; padding: 0.6rem; border-radius: 0.4rem; border: 2px solid #28a745; margin: 0.3rem 0; }
.warning-box { background: #fff3cd; padding: 0.6rem; border-radius: 0.4rem; border-left: 4px solid #ffc107; margin: 0.3rem 0; }
</style>
""", unsafe_allow_html=True)


def load_learned_params():
    if os.path.exists(LEARNED_PARAMS_FILE):
        try:
            with open(LEARNED_PARAMS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'n_corrections': 0, 'gates': {}}


def save_learned_params(params):
    with open(LEARNED_PARAMS_FILE, 'w') as f:
        json.dump(params, f)


def update_learned_params(gate_name, original_polygon, corrected_polygon):
    params = load_learned_params()
    if gate_name not in params['gates']:
        params['gates'][gate_name] = {'avg_adjustment': {'x': 0, 'y': 0}, 'n_samples': 0}
    gate_params = params['gates'][gate_name]
    orig_center = np.mean(original_polygon, axis=0)
    corr_center = np.mean(corrected_polygon, axis=0)
    dx, dy = float(corr_center[0] - orig_center[0]), float(corr_center[1] - orig_center[1])
    gate_params['n_samples'] += 1
    n = gate_params['n_samples']
    alpha = 2 / (n + 1)
    gate_params['avg_adjustment']['x'] = (1 - alpha) * gate_params['avg_adjustment']['x'] + alpha * dx
    gate_params['avg_adjustment']['y'] = (1 - alpha) * gate_params['avg_adjustment']['y'] + alpha * dy
    params['n_corrections'] += 1
    save_learned_params(params)


class FCSReader:
    def __init__(self, fcs_path):
        self.flow_data = flowio.FlowData(fcs_path)
        self.filename = Path(fcs_path).stem
        events = np.array(self.flow_data.events, dtype=np.float64)
        n_ch = self.flow_data.channel_count
        if events.ndim == 1:
            events = events.reshape(-1, n_ch)
        labels = []
        for i in range(1, n_ch + 1):
            pnn = self.flow_data.text.get(f'$P{i}N', '') or self.flow_data.text.get(f'p{i}n', f'Ch{i}')
            labels.append(str(pnn).strip() if pnn else f'Ch{i}')
        self.channels = labels
        self.data = pd.DataFrame(events, columns=labels)


def find_channel(columns, keywords):
    for col in columns:
        for kw in keywords:
            if kw.upper() in col.upper():
                return col
    return None


def biex(x):
    """Transformation biexponentielle"""
    return np.arcsinh(np.asarray(x, float) / 150) * 50


def create_hexagon(center_x, center_y, radius_x, radius_y):
    """Crée un hexagone régulier avec 6 sommets"""
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 0, 60, 120, 180, 240, 300 degrés
    return [(float(center_x + radius_x * np.cos(a)), float(center_y + radius_y * np.sin(a))) for a in angles]


def point_in_polygon(x, y, polygon):
    """Ray casting algorithm pour test point dans polygone"""
    if polygon is None or len(polygon) < 3:
        return np.zeros(len(x), dtype=bool)
    n = len(polygon)
    px = np.array([p[0] for p in polygon])
    py = np.array([p[1] for p in polygon])
    inside = np.zeros(len(x), dtype=bool)
    j = n - 1
    for i in range(n):
        cond = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-10) + px[i])
        inside ^= cond
        j = i
    return inside


def apply_gate(data, x_ch, y_ch, polygon, parent_mask=None):
    """Applique un gate polygonal et retourne le masque booléen"""
    if x_ch is None or y_ch is None or polygon is None or len(polygon) < 3:
        return pd.Series(False, index=data.index)
    
    if parent_mask is not None:
        base = parent_mask.values
    else:
        base = np.ones(len(data), dtype=bool)
    
    x, y = data[x_ch].values, data[y_ch].values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & base
    
    if not valid.any():
        return pd.Series(False, index=data.index)
    
    xt, yt = biex(x), biex(y)
    in_poly = point_in_polygon(xt, yt, polygon)
    
    result = np.zeros(len(data), dtype=bool)
    result[valid & in_poly] = True
    return pd.Series(result, index=data.index)


def find_main_population(xt, yt, mode='main'):
    """
    Trouve la population principale en utilisant GMM avec meilleure sélection.
    Retourne le centre et les rayons pour l'hexagone.
    """
    if len(xt) < 100:
        return None, None, None, None
    
    X = np.column_stack([xt, yt])
    
    # Essayer plusieurs nombres de composants
    best_gmm = None
    best_bic = float('inf')
    
    for n_comp in [2, 3]:
        try:
            gmm = GaussianMixture(n_components=n_comp, covariance_type='full', 
                                  random_state=42, n_init=5, max_iter=200)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except:
            continue
    
    if best_gmm is None:
        return None, None, None, None
    
    labels = best_gmm.predict(X)
    n_comp = best_gmm.n_components
    
    # Sélection du cluster selon le mode
    if mode == 'main':
        # Cluster le plus grand
        sizes = [np.sum(labels == i) for i in range(n_comp)]
        target = np.argmax(sizes)
    elif mode == 'low_x':
        # Cluster avec X le plus bas (Live/Dead négatif)
        means_x = [np.mean(xt[labels == i]) for i in range(n_comp)]
        target = np.argmin(means_x)
    elif mode == 'high_x':
        # Cluster avec X le plus haut
        means_x = [np.mean(xt[labels == i]) for i in range(n_comp)]
        target = np.argmax(means_x)
    elif mode == 'high_x_low_y':
        # CD3+ CD19- : X haut, Y bas
        scores = []
        for i in range(n_comp):
            mx = np.mean(xt[labels == i])
            my = np.mean(yt[labels == i])
            scores.append(mx - my)
        target = np.argmax(scores)
    else:
        target = 0
    
    # Extraire les points du cluster cible
    mask = labels == target
    cx, cy = xt[mask], yt[mask]
    
    if len(cx) < 30:
        return None, None, None, None
    
    # Calculer centre (médiane plus robuste) et rayons
    center_x = np.median(cx)
    center_y = np.median(cy)
    
    # Rayons basés sur les percentiles pour couvrir ~90% des points
    radius_x = np.percentile(np.abs(cx - center_x), 85) * 1.3
    radius_y = np.percentile(np.abs(cy - center_y), 85) * 1.3
    
    # Assurer un minimum de taille
    radius_x = max(radius_x, 5)
    radius_y = max(radius_y, 5)
    
    return center_x, center_y, radius_x, radius_y


def auto_gate_hexagon(data, x_ch, y_ch, parent_mask=None, mode='main'):
    """Auto-gating qui retourne un hexagone bien positionné"""
    if x_ch is None or y_ch is None:
        return None
    
    if parent_mask is not None and parent_mask.sum() > 0:
        subset = data[parent_mask]
    else:
        subset = data
    
    if len(subset) < 100:
        return None
    
    x, y = subset[x_ch].values, subset[y_ch].values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    
    if valid.sum() < 100:
        return None
    
    xt, yt = biex(x[valid]), biex(y[valid])
    
    center_x, center_y, radius_x, radius_y = find_main_population(xt, yt, mode)
    
    if center_x is None:
        return None
    
    return create_hexagon(center_x, center_y, radius_x, radius_y)


def apply_learned_adj(polygon, gate_name):
    """Applique les ajustements appris des corrections précédentes"""
    if polygon is None:
        return None
    params = load_learned_params()
    if gate_name in params['gates']:
        adj = params['gates'][gate_name]['avg_adjustment']
        if abs(adj['x']) > 0.5 or abs(adj['y']) > 0.5:
            return [(p[0] + adj['x'], p[1] + adj['y']) for p in polygon]
    return polygon


def find_closest_point(polygon, click_x, click_y, threshold=12):
    """Trouve le sommet le plus proche du clic"""
    if polygon is None:
        return None
    min_dist = float('inf')
    closest_idx = None
    for i, (px, py) in enumerate(polygon):
        dist = np.sqrt((px - click_x)**2 + (py - click_y)**2)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            closest_idx = i
    return closest_idx


def create_plot(data, x_ch, y_ch, x_label, y_label, title, polygon, parent_mask, gate_name, selected_point=None):
    """Crée le graphique Plotly avec hexagone et points éditables"""
    
    if x_ch is None or y_ch is None:
        fig = go.Figure()
        fig.add_annotation(text="Canal non trouvé", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, 0, 0.0
    
    if parent_mask is not None and parent_mask.sum() > 0:
        subset = data[parent_mask]
    else:
        subset = data
    
    n_parent = len(subset)
    
    if n_parent == 0:
        fig = go.Figure()
        fig.add_annotation(text="Pas de données", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, 0, 0.0
    
    x, y = subset[x_ch].values, subset[y_ch].values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xt, yt = biex(x[valid]), biex(y[valid])
    
    # Sous-échantillonner pour affichage
    max_pts = 8000
    if len(xt) > max_pts:
        idx = np.random.choice(len(xt), max_pts, replace=False)
        xd, yd = xt[idx], yt[idx]
    else:
        xd, yd = xt, yt
    
    fig = go.Figure()
    
    # Scatter plot des données
    fig.add_trace(go.Scattergl(
        x=xd, y=yd,
        mode='markers',
        marker=dict(size=2, color=yd, colorscale='Viridis', opacity=0.6),
        hoverinfo='skip',
        name='Data'
    ))
    
    n_in, pct = 0, 0.0
    
    if polygon and len(polygon) >= 3:
        # Calculer stats sur toutes les données
        full_mask = apply_gate(data, x_ch, y_ch, polygon, parent_mask)
        n_in = int(full_mask.sum())
        pct = float(n_in / n_parent * 100) if n_parent > 0 else 0.0
        
        # Tracer l'hexagone
        px = [p[0] for p in polygon] + [polygon[0][0]]
        py = [p[1] for p in polygon] + [polygon[0][1]]
        
        # Remplissage
        fig.add_trace(go.Scatter(
            x=px, y=py,
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.15)',
            line=dict(color='red', width=2.5),
            mode='lines',
            name='Gate',
            hoverinfo='skip'
        ))
        
        # Points de contrôle (sommets)
        point_colors = []
        point_sizes = []
        for i in range(len(polygon)):
            if selected_point is not None and i == selected_point:
                point_colors.append('lime')
                point_sizes.append(22)
            else:
                point_colors.append('red')
                point_sizes.append(16)
        
        fig.add_trace(go.Scatter(
            x=[p[0] for p in polygon],
            y=[p[1] for p in polygon],
            mode='markers+text',
            marker=dict(
                size=point_sizes, 
                color=point_colors, 
                symbol='circle',
                line=dict(color='darkred', width=2)
            ),
            text=[str(i+1) for i in range(len(polygon))],
            textposition='top center',
            textfont=dict(size=12, color='darkred', family='Arial Black'),
            name='Sommets',
            hovertemplate='<b>Point %{text}</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
        ))
        
        # Annotation centrale
        cx = np.mean([p[0] for p in polygon])
        cy = np.mean([p[1] for p in polygon])
        fig.add_annotation(
            x=cx, y=cy,
            text=f"<b>{gate_name}</b><br>{pct:.1f}%<br>({n_in:,})",
            showarrow=False,
            font=dict(size=12, color='darkred'),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='red',
            borderwidth=1,
            borderpad=4
        )
    
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sup>Parent: {n_parent:,}</sup>", x=0.5, font=dict(size=13)),
        xaxis_title=f"<b>{x_label}</b>",
        yaxis_title=f"<b>{y_label}</b>",
        showlegend=False,
        height=420,
        margin=dict(l=60, r=30, t=70, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#eee', zeroline=False, showline=True, linecolor='#999', mirror=True),
        yaxis=dict(showgrid=True, gridcolor='#eee', zeroline=False, showline=True, linecolor='#999', mirror=True),
    )
    
    return fig, n_in, pct


def move_polygon(polygon, dx, dy):
    if polygon is None:
        return None
    return [(p[0] + dx, p[1] + dy) for p in polygon]


def scale_polygon(polygon, factor):
    if polygon is None:
        return None
    center = np.mean(polygon, axis=0)
    return [(float(center[0] + factor * (p[0] - center[0])), float(center[1] + factor * (p[1] - center[1]))) for p in polygon]


# ==================== MAIN ====================

st.markdown('<h1 class="main-header">🔬 FACS - Hexagones Interactifs</h1>', unsafe_allow_html=True)

learned = load_learned_params()
n_learned = learned.get('n_corrections', 0)

if n_learned > 0:
    st.success(f"🧠 {n_learned} correction(s) apprises - L'auto-gating s'améliore!")

if not HAS_PLOTLY_EVENTS:
    st.warning("⚠️ Module `streamlit-plotly-events` non installé. Ajoutez-le dans requirements.txt pour l'édition par clic.")

# Session state
if 'reader' not in st.session_state:
    st.session_state.reader = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'channels' not in st.session_state:
    st.session_state.channels = {}
if 'polygons' not in st.session_state:
    st.session_state.polygons = {}
if 'original_polygons' not in st.session_state:
    st.session_state.original_polygons = {}
if 'auto_done' not in st.session_state:
    st.session_state.auto_done = False
if 'selected_gate' not in st.session_state:
    st.session_state.selected_gate = None
if 'selected_point' not in st.session_state:
    st.session_state.selected_point = None

# Upload
uploaded = st.file_uploader("📁 Fichier FCS", type=['fcs'])

if uploaded:
    if st.session_state.reader is None or st.session_state.get('fname') != uploaded.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        
        with st.spinner("Chargement du fichier FCS..."):
            reader = FCSReader(tmp_path)
            st.session_state.reader = reader
            st.session_state.data = reader.data
            st.session_state.fname = uploaded.name
            st.session_state.polygons = {}
            st.session_state.original_polygons = {}
            st.session_state.auto_done = False
            st.session_state.selected_gate = None
            st.session_state.selected_point = None
            
            cols = reader.data.columns
            st.session_state.channels = {
                'FSC-A': find_channel(cols, ['FSC-A']),
                'FSC-H': find_channel(cols, ['FSC-H']),
                'SSC-A': find_channel(cols, ['SSC-A']),
                'LiveDead': find_channel(cols, ['LiveDead', 'Viab', 'Aqua', 'Live', 'L/D']),
                'hCD45': find_channel(cols, ['PerCP', 'CD45', 'hCD45']),
                'CD3': find_channel(cols, ['AF488', 'FITC', 'CD3']),
                'CD19': find_channel(cols, ['PE-Fire', 'CD19', 'PE-Cy7']),
                'CD4': find_channel(cols, ['BV650', 'CD4', 'APC']),
                'CD8': find_channel(cols, ['BUV805', 'CD8', 'APC-Cy7', 'PerCP-Cy5']),
            }

    reader = st.session_state.reader
    data = st.session_state.data
    ch = st.session_state.channels
    n_total = len(data)
    
    # Métriques
    c1, c2, c3 = st.columns(3)
    c1.metric("Événements", f"{n_total:,}")
    c2.metric("Canaux", len(reader.channels))
    c3.metric("Fichier", reader.filename[:30])
    
    with st.expander("📋 Canaux détectés", expanded=False):
        col1, col2 = st.columns(2)
        items = list(ch.items())
        for i, (name, canal) in enumerate(items):
            with col1 if i < len(items)//2 else col2:
                st.write(f"{'✅' if canal else '❌'} **{name}**: {canal or 'Non trouvé'}")
    
    if ch['FSC-A'] is None or ch['SSC-A'] is None:
        st.error("❌ Canaux FSC-A ou SSC-A non trouvés!")
        st.stop()
    
    st.markdown("---")
    
    # ==================== AUTO-GATING ====================
    
    if not st.session_state.auto_done:
        if st.button("🚀 LANCER L'AUTO-GATING", type="primary", use_container_width=True):
            prog = st.progress(0, "Initialisation...")
            
            # 1. Cells (FSC-A vs SSC-A)
            prog.progress(10, "Gate Cells...")
            poly = auto_gate_hexagon(data, ch['FSC-A'], ch['SSC-A'], None, 'main')
            poly = apply_learned_adj(poly, 'cells')
            st.session_state.polygons['cells'] = poly
            st.session_state.original_polygons['cells'] = list(poly) if poly else None
            
            # 2. Singlets (FSC-A vs FSC-H)
            prog.progress(30, "Gate Singlets...")
            if ch['FSC-H'] and poly:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], poly, None)
                poly2 = auto_gate_hexagon(data, ch['FSC-A'], ch['FSC-H'], cells_m, 'main')
                poly2 = apply_learned_adj(poly2, 'singlets')
            else:
                poly2 = None
            st.session_state.polygons['singlets'] = poly2
            st.session_state.original_polygons['singlets'] = list(poly2) if poly2 else None
            
            # 3. Live (LiveDead vs SSC-A) - population NEGATIVE (low_x)
            prog.progress(50, "Gate Live...")
            if ch['LiveDead']:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], st.session_state.polygons['cells'], None)
                sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], st.session_state.polygons['singlets'], cells_m) if st.session_state.polygons['singlets'] else cells_m
                poly3 = auto_gate_hexagon(data, ch['LiveDead'], ch['SSC-A'], sing_m, 'low_x')
                poly3 = apply_learned_adj(poly3, 'live')
            else:
                poly3 = None
            st.session_state.polygons['live'] = poly3
            st.session_state.original_polygons['live'] = list(poly3) if poly3 else None
            
            # 4. hCD45+ (hCD45 vs SSC-A) - population POSITIVE (high_x)
            prog.progress(75, "Gate hCD45+...")
            if ch['hCD45']:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], st.session_state.polygons['cells'], None)
                sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], st.session_state.polygons['singlets'], cells_m) if st.session_state.polygons['singlets'] else cells_m
                live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'], st.session_state.polygons['live'], sing_m) if st.session_state.polygons['live'] else sing_m
                poly4 = auto_gate_hexagon(data, ch['hCD45'], ch['SSC-A'], live_m, 'high_x')
                poly4 = apply_learned_adj(poly4, 'hcd45')
            else:
                poly4 = None
            st.session_state.polygons['hcd45'] = poly4
            st.session_state.original_polygons['hcd45'] = list(poly4) if poly4 else None
            
            prog.progress(100, "Terminé!")
            st.session_state.auto_done = True
            st.rerun()
    
    # ==================== AFFICHAGE ET ÉDITION ====================
    
    if st.session_state.auto_done:
        polygons = st.session_state.polygons
        
        # Instructions
        if HAS_PLOTLY_EVENTS:
            st.markdown("""
            <div class="info-box">
            <b>✏️ Édition directe:</b> Cliquez sur un <b>sommet rouge</b> pour le sélectionner (devient vert), 
            puis cliquez sur la <b>nouvelle position</b> dans le graphique.
            </div>
            """, unsafe_allow_html=True)
        
        # Afficher sélection active
        if st.session_state.selected_gate and st.session_state.selected_point is not None:
            st.markdown(f"""
            <div class="selected-box">
            ✅ <b>Sélectionné:</b> {st.session_state.selected_gate.upper()} - Point {st.session_state.selected_point + 1}
            → Cliquez sur la nouvelle position
            </div>
            """, unsafe_allow_html=True)
            if st.button("❌ Annuler sélection"):
                st.session_state.selected_gate = None
                st.session_state.selected_point = None
                st.rerun()
        
        # Recalcul cascade
        cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], polygons.get('cells'), None)
        sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], polygons.get('singlets'), cells_m) if polygons.get('singlets') else cells_m
        live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'], polygons.get('live'), sing_m) if polygons.get('live') else sing_m
        hcd45_m = apply_gate(data, ch['hCD45'], ch['SSC-A'], polygons.get('hcd45'), live_m) if polygons.get('hcd45') else live_m
        
        # Config gates
        gates_config = [
            ('cells', 'Cells', ch['FSC-A'], ch['SSC-A'], 'FSC-A', 'SSC-A', 'Ungated → Cells', None),
            ('singlets', 'Singlets', ch['FSC-A'], ch['FSC-H'], 'FSC-A', 'FSC-H', 'Cells → Singlets', cells_m),
            ('live', 'Live', ch['LiveDead'], ch['SSC-A'], 'Live/Dead', 'SSC-A', 'Singlets → Live', sing_m),
            ('hcd45', 'hCD45+', ch['hCD45'], ch['SSC-A'], 'hCD45', 'SSC-A', 'Live → hCD45+', live_m),
        ]
        
        stats = []
        
        # Affichage 2x2
        for row in range(2):
            cols_display = st.columns(2)
            for col_idx in range(2):
                gate_idx = row * 2 + col_idx
                if gate_idx >= len(gates_config):
                    break
                
                gkey, gname, x_ch, y_ch, x_label, y_label, title, parent_mask = gates_config[gate_idx]
                
                if x_ch is None or y_ch is None:
                    continue
                
                with cols_display[col_idx]:
                    st.markdown(f"#### {gname}")
                    
                    # Point sélectionné pour ce gate?
                    sel_pt = None
                    if st.session_state.selected_gate == gkey:
                        sel_pt = st.session_state.selected_point
                    
                    # Créer graphique
                    fig, n_in, pct = create_plot(
                        data, x_ch, y_ch, x_label, y_label, title,
                        polygons.get(gkey), parent_mask, gname, sel_pt
                    )
                    
                    # Affichage avec ou sans événements de clic
                    if HAS_PLOTLY_EVENTS:
                        clicked = plotly_events(fig, click_event=True, key=f"plot_{gkey}_{row}_{col_idx}")
                        
                        # Traiter clics
                        if clicked and len(clicked) > 0:
                            click_x = clicked[0].get('x')
                            click_y = clicked[0].get('y')
                            
                            if click_x is not None and click_y is not None:
                                poly = polygons.get(gkey)
                                
                                if poly:
                                    # Vérifier si clic sur un sommet
                                    closest = find_closest_point(poly, click_x, click_y, threshold=15)
                                    
                                    if closest is not None:
                                        # Sélectionner ce point
                                        st.session_state.selected_gate = gkey
                                        st.session_state.selected_point = closest
                                        st.rerun()
                                    elif st.session_state.selected_gate == gkey and st.session_state.selected_point is not None:
                                        # Déplacer le point sélectionné
                                        new_poly = list(poly)
                                        new_poly[st.session_state.selected_point] = (click_x, click_y)
                                        st.session_state.polygons[gkey] = new_poly
                                        st.session_state.selected_gate = None
                                        st.session_state.selected_point = None
                                        st.rerun()
                    else:
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{gkey}")
                    
                    parent_name = title.split('→')[0].strip() if '→' in title else 'Ungated'
                    stats.append((gname, parent_name, n_in, pct))
                    
                    # Contrôles manuels
                    poly = polygons.get(gkey)
                    if poly:
                        with st.expander(f"🎛️ Contrôles {gname}", expanded=False):
                            # Édition point par point
                            st.markdown("**Modifier un point:**")
                            c1, c2 = st.columns(2)
                            with c1:
                                pt_idx = st.selectbox("Point", range(6), format_func=lambda x: f"Point {x+1}", key=f"pt_{gkey}")
                            with c2:
                                st.caption(f"({poly[pt_idx][0]:.1f}, {poly[pt_idx][1]:.1f})")
                            
                            cx, cy = st.columns(2)
                            with cx:
                                new_x = st.number_input("X", value=float(poly[pt_idx][0]), step=2.0, key=f"x_{gkey}")
                            with cy:
                                new_y = st.number_input("Y", value=float(poly[pt_idx][1]), step=2.0, key=f"y_{gkey}")
                            
                            if st.button("✅ Appliquer", key=f"apply_{gkey}", use_container_width=True):
                                new_poly = list(poly)
                                new_poly[pt_idx] = (new_x, new_y)
                                st.session_state.polygons[gkey] = new_poly
                                st.rerun()
                            
                            st.markdown("---")
                            st.markdown("**Déplacer tout:**")
                            step = st.slider("Pas", 2, 30, 10, key=f"step_{gkey}")
                            
                            b1, b2, b3, b4 = st.columns(4)
                            with b1:
                                if st.button("⬆️", key=f"up_{gkey}"):
                                    st.session_state.polygons[gkey] = move_polygon(poly, 0, step)
                                    st.rerun()
                            with b2:
                                if st.button("⬇️", key=f"dn_{gkey}"):
                                    st.session_state.polygons[gkey] = move_polygon(poly, 0, -step)
                                    st.rerun()
                            with b3:
                                if st.button("⬅️", key=f"lt_{gkey}"):
                                    st.session_state.polygons[gkey] = move_polygon(poly, -step, 0)
                                    st.rerun()
                            with b4:
                                if st.button("➡️", key=f"rt_{gkey}"):
                                    st.session_state.polygons[gkey] = move_polygon(poly, step, 0)
                                    st.rerun()
                            
                            b5, b6 = st.columns(2)
                            with b5:
                                if st.button("➕ Agrandir", key=f"grow_{gkey}", use_container_width=True):
                                    st.session_state.polygons[gkey] = scale_polygon(poly, 1.15)
                                    st.rerun()
                            with b6:
                                if st.button("➖ Réduire", key=f"shrink_{gkey}", use_container_width=True):
                                    st.session_state.polygons[gkey] = scale_polygon(poly, 0.85)
                                    st.rerun()
        
        st.markdown("---")
        
        # Actions globales
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("💾 Sauvegarder corrections", type="primary", use_container_width=True):
                n_saved = 0
                for gname in polygons:
                    curr = polygons.get(gname)
                    orig = st.session_state.original_polygons.get(gname)
                    if curr and orig:
                        if str(curr) != str(orig):
                            update_learned_params(gname, orig, curr)
                            n_saved += 1
                if n_saved:
                    st.success(f"✅ {n_saved} correction(s) sauvegardée(s)!")
                else:
                    st.info("Aucune modification détectée")
        
        with col_b:
            if st.button("🔃 Réinitialiser", use_container_width=True):
                st.session_state.polygons = {k: list(v) if v else None for k, v in st.session_state.original_polygons.items()}
                st.session_state.selected_gate = None
                st.session_state.selected_point = None
                st.rerun()
        
        with col_c:
            if st.button("🔄 Rafraîchir", use_container_width=True):
                st.rerun()
        
        # Résumé
        st.markdown("### 📊 Résumé des Populations")
        df = pd.DataFrame(stats, columns=['Population', 'Parent', 'Count', '% Parent'])
        df['% Total'] = (df['Count'] / n_total * 100).round(2)
        df['% Parent'] = df['% Parent'].round(1)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export
        c1, c2 = st.columns(2)
        c1.download_button("📥 CSV", df.to_csv(index=False), f"{reader.filename}_stats.csv", "text/csv", use_container_width=True)
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        buf.seek(0)
        c2.download_button("📥 Excel", buf, f"{reader.filename}_stats.xlsx", use_container_width=True)

else:
    st.markdown("""
    <div class="info-box">
    <h3>🔬 FACS Auto-Gating avec Hexagones</h3>
    <ul>
    <li><b>Auto-gating GMM</b> optimisé pour chaque population</li>
    <li><b>Gates hexagonaux</b> (6 sommets) ajustables</li>
    <li><b>Édition par clic</b> sur les sommets du graphique</li>
    <li><b>Mise à jour cascade</b> automatique</li>
    <li><b>Apprentissage</b> des corrections</li>
    </ul>
    <p>👉 Uploadez un fichier FCS pour commencer</p>
    </div>
    """, unsafe_allow_html=True)

st.caption(f"🔬 FACS Hexagones v3 | 🧠 {n_learned} corrections | {'✅ Clic activé' if HAS_PLOTLY_EVENTS else '⚠️ Clic désactivé'}")
