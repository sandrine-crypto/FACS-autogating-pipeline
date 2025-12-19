#!/usr/bin/env python3
"""
FACS Autogating - Gates Hexagonaux avec Édition Directe sur Graphique
- Gates hexagonaux (6 sommets)
- Cliquer sur un sommet pour le sélectionner, puis cliquer sur la nouvelle position
- Mise à jour en cascade de toutes les populations
- Auto-gating GMM + apprentissage
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull
from pathlib import Path
import tempfile
import io
import json
import os
from datetime import datetime
import flowio

st.set_page_config(page_title="FACS - Hexagones Interactifs", page_icon="🔬", layout="wide")

LEARNED_PARAMS_FILE = "learned_gating_params.json"

st.markdown("""
<style>
.main-header { font-size: 1.8rem; color: #2c3e50; text-align: center; margin-bottom: 0.5rem; }
.info-box { background: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid #0066cc; margin: 0.5rem 0; }
.selected-info { background: #d4edda; padding: 0.5rem; border-radius: 0.3rem; border: 1px solid #28a745; margin: 0.3rem 0; }
.warning-info { background: #fff3cd; padding: 0.5rem; border-radius: 0.3rem; border: 1px solid #ffc107; margin: 0.3rem 0; }
.stats-card { background: #f8f9fa; padding: 0.8rem; border-radius: 0.5rem; border: 1px solid #dee2e6; }
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
    return np.arcsinh(np.asarray(x, float) / 150) * 50


def create_hexagon(center_x, center_y, radius_x, radius_y):
    """Crée un hexagone régulier"""
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points
    return [(center_x + radius_x * np.cos(a), center_y + radius_y * np.sin(a)) for a in angles]


def point_in_polygon(x, y, polygon):
    """Ray casting algorithm"""
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
    if x_ch is None or y_ch is None or polygon is None or len(polygon) < 3:
        return pd.Series(False, index=data.index)
    base = parent_mask.values if parent_mask is not None else np.ones(len(data), dtype=bool)
    x, y = data[x_ch].values, data[y_ch].values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & base
    if not valid.any():
        return pd.Series(False, index=data.index)
    xt, yt = biex(x), biex(y)
    in_poly = point_in_polygon(xt, yt, polygon)
    result = np.zeros(len(data), dtype=bool)
    result[valid & in_poly] = True
    return pd.Series(result, index=data.index)


def auto_gate_gmm_hexagon(data, x_ch, y_ch, parent_mask=None, n_comp=2, mode='main'):
    """Auto-gating avec GMM, retourne un hexagone"""
    if x_ch is None or y_ch is None:
        return None
    subset = data[parent_mask] if parent_mask is not None and parent_mask.sum() > 0 else data
    if len(subset) < 100:
        return None
    x, y = subset[x_ch].values, subset[y_ch].values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if valid.sum() < 100:
        return None
    xt, yt = biex(x[valid]), biex(y[valid])
    X = np.column_stack([xt, yt])
    try:
        gmm = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42, n_init=3)
        gmm.fit(X)
        labels = gmm.predict(X)
        if mode == 'main':
            target = np.argmax([np.sum(labels == i) for i in range(n_comp)])
        elif mode == 'low_x':
            target = np.argmin([np.mean(xt[labels == i]) for i in range(n_comp)])
        elif mode == 'high_x':
            target = np.argmax([np.mean(xt[labels == i]) for i in range(n_comp)])
        elif mode == 'high_x_low_y':
            target = np.argmax([np.mean(xt[labels == i]) - np.mean(yt[labels == i]) for i in range(n_comp)])
        else:
            target = 0
        mask = labels == target
        cx, cy = xt[mask], yt[mask]
        if len(cx) < 30:
            return None
        # Créer hexagone centré sur le cluster
        center_x, center_y = np.median(cx), np.median(cy)
        radius_x = np.percentile(np.abs(cx - center_x), 90) * 1.2
        radius_y = np.percentile(np.abs(cy - center_y), 90) * 1.2
        return create_hexagon(center_x, center_y, radius_x, radius_y)
    except:
        return None


def apply_learned_adj(polygon, gate_name):
    if polygon is None:
        return None
    params = load_learned_params()
    if gate_name in params['gates']:
        adj = params['gates'][gate_name]['avg_adjustment']
        if abs(adj['x']) > 0.5 or abs(adj['y']) > 0.5:
            return [(p[0] + adj['x'], p[1] + adj['y']) for p in polygon]
    return polygon


def find_closest_point(polygon, click_x, click_y, threshold=15):
    """Trouve le point le plus proche du clic"""
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


def create_interactive_hexagon_plot(data, x_ch, y_ch, x_label, y_label, title, 
                                     polygon, parent_mask, gate_name, selected_point=None):
    """Crée un graphique Plotly avec hexagone et points cliquables"""
    
    if x_ch is None or y_ch is None:
        fig = go.Figure()
        fig.add_annotation(text="Canal non trouvé", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, 0, 0
    
    subset = data[parent_mask] if parent_mask is not None and parent_mask.sum() > 0 else data
    n_parent = len(subset)
    
    if n_parent == 0:
        fig = go.Figure()
        fig.add_annotation(text="Pas de données", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig, 0, 0
    
    x, y = subset[x_ch].values, subset[y_ch].values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xt, yt = biex(x[valid]), biex(y[valid])
    
    # Sous-échantillonner
    if len(xt) > 6000:
        idx = np.random.choice(len(xt), 6000, replace=False)
        xd, yd = xt[idx], yt[idx]
    else:
        xd, yd = xt, yt
    
    fig = go.Figure()
    
    # Scatter plot des données
    fig.add_trace(go.Scattergl(
        x=xd, y=yd,
        mode='markers',
        marker=dict(size=2, color=yd, colorscale='Viridis', opacity=0.5),
        hoverinfo='skip',
        name='Data'
    ))
    
    n_in, pct = 0, 0
    
    if polygon and len(polygon) >= 3:
        # Calculer stats
        full_mask = apply_gate(data, x_ch, y_ch, polygon, parent_mask)
        n_in = full_mask.sum()
        pct = n_in / n_parent * 100 if n_parent > 0 else 0
        
        # Hexagone - remplissage
        px = [p[0] for p in polygon] + [polygon[0][0]]
        py = [p[1] for p in polygon] + [polygon[0][1]]
        
        fig.add_trace(go.Scatter(
            x=px, y=py,
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='red', width=2),
            mode='lines',
            name='Gate',
            hoverinfo='skip'
        ))
        
        # Points de contrôle (sommets de l'hexagone)
        point_colors = ['red'] * len(polygon)
        point_sizes = [14] * len(polygon)
        if selected_point is not None and selected_point < len(polygon):
            point_colors[selected_point] = 'lime'
            point_sizes[selected_point] = 20
        
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
            textfont=dict(size=11, color='darkred', family='Arial Black'),
            name='Sommets',
            hovertemplate='<b>Point %{text}</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br><i>Cliquez pour sélectionner</i><extra></extra>',
            customdata=list(range(len(polygon)))
        ))
        
        # Annotation centrale
        cx = np.mean([p[0] for p in polygon])
        cy = np.mean([p[1] for p in polygon])
        fig.add_annotation(
            x=cx, y=cy,
            text=f"<b>{gate_name}</b><br>{pct:.1f}%<br>({n_in:,})",
            showarrow=False,
            font=dict(size=11, color='darkred'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='red',
            borderwidth=1
        )
    
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sup>n={n_parent:,}</sup>", x=0.5, font=dict(size=12)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False,
        height=400,
        margin=dict(l=50, r=20, t=60, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False, showline=True, linecolor='#ccc'),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False, showline=True, linecolor='#ccc'),
        dragmode='pan'
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
    return [(center[0] + factor * (p[0] - center[0]), center[1] + factor * (p[1] - center[1])) for p in polygon]


# ===== MAIN =====
st.markdown('<h1 class="main-header">🔬 FACS - Hexagones Interactifs</h1>', unsafe_allow_html=True)

learned = load_learned_params()
n_learned = learned.get('n_corrections', 0)
if n_learned > 0:
    st.info(f"🧠 {n_learned} correction(s) apprises")

# Session state
for key in ['reader', 'data', 'channels', 'polygons', 'original_polygons', 'auto_done']:
    if key not in st.session_state:
        st.session_state[key] = {} if 'polygon' in key or key == 'channels' else None

if st.session_state.get('auto_done') is None:
    st.session_state.auto_done = False

# État de sélection pour l'édition
if 'selected_gate' not in st.session_state:
    st.session_state.selected_gate = None
if 'selected_point_idx' not in st.session_state:
    st.session_state.selected_point_idx = None

uploaded = st.file_uploader("📁 Fichier FCS", type=['fcs'])

if uploaded:
    if st.session_state.reader is None or st.session_state.get('fname') != uploaded.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        with st.spinner("Chargement..."):
            reader = FCSReader(tmp_path)
            st.session_state.reader = reader
            st.session_state.data = reader.data
            st.session_state.fname = uploaded.name
            st.session_state.polygons = {}
            st.session_state.original_polygons = {}
            st.session_state.auto_done = False
            st.session_state.selected_gate = None
            st.session_state.selected_point_idx = None
            cols = reader.data.columns
            st.session_state.channels = {
                'FSC-A': find_channel(cols, ['FSC-A', 'FSC']),
                'FSC-H': find_channel(cols, ['FSC-H']),
                'SSC-A': find_channel(cols, ['SSC-A', 'SSC']),
                'LiveDead': find_channel(cols, ['LiveDead', 'Viab', 'Aqua', 'Live']),
                'hCD45': find_channel(cols, ['PerCP', 'CD45']),
                'CD3': find_channel(cols, ['AF488', 'FITC', 'CD3']),
                'CD19': find_channel(cols, ['PE-Fire', 'CD19']),
                'CD4': find_channel(cols, ['BV650', 'CD4']),
                'CD8': find_channel(cols, ['BUV805', 'APC-Cy7', 'CD8']),
            }

    reader = st.session_state.reader
    data = st.session_state.data
    ch = st.session_state.channels
    n_total = len(data)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Événements", f"{n_total:,}")
    c2.metric("Canaux", len(reader.channels))
    c3.metric("Fichier", reader.filename[:25])
    
    with st.expander("📋 Canaux détectés"):
        for name, canal in ch.items():
            st.write(f"{'✅' if canal else '❌'} **{name}**: {canal or 'Non trouvé'}")
    
    if ch['FSC-A'] is None or ch['SSC-A'] is None:
        st.error("❌ FSC-A ou SSC-A non trouvé!")
        st.stop()
    
    st.markdown("---")
    
    # AUTO-GATING
    if not st.session_state.auto_done:
        if st.button("🚀 LANCER L'AUTO-GATING", type="primary", use_container_width=True):
            prog = st.progress(0)
            
            # Cells
            poly = auto_gate_gmm_hexagon(data, ch['FSC-A'], ch['SSC-A'], None, 2, 'main')
            poly = apply_learned_adj(poly, 'cells')
            st.session_state.polygons['cells'] = poly
            st.session_state.original_polygons['cells'] = list(poly) if poly else None
            prog.progress(25)
            
            # Singlets
            if ch['FSC-H']:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], poly, None)
                poly = auto_gate_gmm_hexagon(data, ch['FSC-A'], ch['FSC-H'], cells_m, 2, 'main')
                poly = apply_learned_adj(poly, 'singlets')
            else:
                poly = None
            st.session_state.polygons['singlets'] = poly
            st.session_state.original_polygons['singlets'] = list(poly) if poly else None
            prog.progress(50)
            
            # Live
            if ch['LiveDead']:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], st.session_state.polygons['cells'], None)
                sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], st.session_state.polygons['singlets'], cells_m) if st.session_state.polygons['singlets'] else cells_m
                poly = auto_gate_gmm_hexagon(data, ch['LiveDead'], ch['SSC-A'], sing_m, 2, 'low_x')
                poly = apply_learned_adj(poly, 'live')
            else:
                poly = None
            st.session_state.polygons['live'] = poly
            st.session_state.original_polygons['live'] = list(poly) if poly else None
            prog.progress(75)
            
            # hCD45
            if ch['hCD45']:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], st.session_state.polygons['cells'], None)
                sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], st.session_state.polygons['singlets'], cells_m) if st.session_state.polygons['singlets'] else cells_m
                live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'], st.session_state.polygons['live'], sing_m) if st.session_state.polygons['live'] else sing_m
                poly = auto_gate_gmm_hexagon(data, ch['hCD45'], ch['SSC-A'], live_m, 2, 'high_x')
                poly = apply_learned_adj(poly, 'hcd45')
            else:
                poly = None
            st.session_state.polygons['hcd45'] = poly
            st.session_state.original_polygons['hcd45'] = list(poly) if poly else None
            prog.progress(100)
            
            st.session_state.auto_done = True
            st.rerun()
    
    # AFFICHAGE ET ÉDITION
    if st.session_state.auto_done:
        polygons = st.session_state.polygons
        
        # Instructions
        st.markdown("""
        <div class="info-box">
        <b>📌 Modification directe sur le graphique:</b><br>
        1️⃣ <b>Cliquez sur un sommet</b> (point rouge) pour le sélectionner (devient vert)<br>
        2️⃣ <b>Cliquez sur la nouvelle position</b> dans le graphique<br>
        3️⃣ Les statistiques se mettent à jour automatiquement
        </div>
        """, unsafe_allow_html=True)
        
        # Afficher le point sélectionné
        if st.session_state.selected_gate and st.session_state.selected_point_idx is not None:
            st.markdown(f"""
            <div class="selected-info">
            ✅ <b>Point sélectionné:</b> Gate <b>{st.session_state.selected_gate}</b>, 
            Sommet <b>{st.session_state.selected_point_idx + 1}</b> — 
            Cliquez sur la nouvelle position ou sélectionnez un autre point
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("❌ Désélectionner", key="deselect"):
                st.session_state.selected_gate = None
                st.session_state.selected_point_idx = None
                st.rerun()
        
        # Recalcul des masques (cascade complète)
        cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], polygons.get('cells'), None)
        sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], polygons.get('singlets'), cells_m) if polygons.get('singlets') else cells_m
        live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'], polygons.get('live'), sing_m) if polygons.get('live') else sing_m
        hcd45_m = apply_gate(data, ch['hCD45'], ch['SSC-A'], polygons.get('hcd45'), live_m) if polygons.get('hcd45') else live_m
        
        # Configuration des gates
        gates_config = [
            ('cells', 'Cells', ch['FSC-A'], ch['SSC-A'], 'FSC-A', 'SSC-A', 'Ungated → Cells', None),
            ('singlets', 'Singlets', ch['FSC-A'], ch['FSC-H'], 'FSC-A', 'FSC-H', 'Cells → Singlets', cells_m),
            ('live', 'Live', ch['LiveDead'], ch['SSC-A'], 'Live/Dead', 'SSC-A', 'Singlets → Live', sing_m),
            ('hcd45', 'hCD45+', ch['hCD45'], ch['SSC-A'], 'hCD45', 'SSC-A', 'Live → hCD45+', live_m),
        ]
        
        stats = []
        
        # Affichage en 2x2
        for row in range(2):
            cols_display = st.columns(2)
            for col_idx in range(2):
                gate_idx = row * 2 + col_idx
                if gate_idx >= len(gates_config):
                    break
                
                gkey, gname, x_ch, y_ch, x_label, y_label, title, parent_mask = gates_config[gate_idx]
                
                if polygons.get(gkey) is None and gkey != 'cells':
                    continue
                
                with cols_display[col_idx]:
                    st.markdown(f"#### {gate_idx + 1}️⃣ {gname}")
                    
                    # Déterminer si ce gate a un point sélectionné
                    selected_pt = None
                    if st.session_state.selected_gate == gkey:
                        selected_pt = st.session_state.selected_point_idx
                    
                    # Créer le graphique
                    fig, n_in, pct = create_interactive_hexagon_plot(
                        data, x_ch, y_ch, x_label, y_label, title,
                        polygons.get(gkey), parent_mask, gname, selected_pt
                    )
                    
                    # Afficher avec capture des clics
                    clicked = plotly_events(fig, click_event=True, key=f"plot_{gkey}")
                    
                    stats.append((gname, title.split('→')[0].strip() if '→' in title else 'Ungated', n_in, pct))
                    
                    # Traiter les clics
                    if clicked and len(clicked) > 0:
                        click_x = clicked[0].get('x')
                        click_y = clicked[0].get('y')
                        
                        if click_x is not None and click_y is not None:
                            poly = polygons.get(gkey)
                            
                            # Vérifier si on clique sur un sommet
                            closest_pt = find_closest_point(poly, click_x, click_y, threshold=20)
                            
                            if closest_pt is not None:
                                # Sélection d'un point
                                st.session_state.selected_gate = gkey
                                st.session_state.selected_point_idx = closest_pt
                                st.rerun()
                            elif st.session_state.selected_gate == gkey and st.session_state.selected_point_idx is not None:
                                # Déplacer le point sélectionné
                                pt_idx = st.session_state.selected_point_idx
                                new_poly = list(poly)
                                new_poly[pt_idx] = (click_x, click_y)
                                st.session_state.polygons[gkey] = new_poly
                                # Désélectionner après déplacement
                                st.session_state.selected_gate = None
                                st.session_state.selected_point_idx = None
                                st.rerun()
                    
                    # Boutons de contrôle rapide
                    with st.expander("🎛️ Contrôles", expanded=False):
                        poly = polygons.get(gkey)
                        if poly:
                            move_step = st.slider("Pas", 1, 20, 5, key=f"step_{gkey}")
                            
                            c1, c2, c3, c4 = st.columns(4)
                            with c1:
                                if st.button("⬆️", key=f"up_{gkey}"):
                                    st.session_state.polygons[gkey] = move_polygon(poly, 0, move_step)
                                    st.rerun()
                            with c2:
                                if st.button("⬇️", key=f"dn_{gkey}"):
                                    st.session_state.polygons[gkey] = move_polygon(poly, 0, -move_step)
                                    st.rerun()
                            with c3:
                                if st.button("⬅️", key=f"lt_{gkey}"):
                                    st.session_state.polygons[gkey] = move_polygon(poly, -move_step, 0)
                                    st.rerun()
                            with c4:
                                if st.button("➡️", key=f"rt_{gkey}"):
                                    st.session_state.polygons[gkey] = move_polygon(poly, move_step, 0)
                                    st.rerun()
                            
                            c5, c6 = st.columns(2)
                            with c5:
                                if st.button("➕ Agrandir", key=f"grow_{gkey}"):
                                    st.session_state.polygons[gkey] = scale_polygon(poly, 1.1)
                                    st.rerun()
                            with c6:
                                if st.button("➖ Réduire", key=f"shrink_{gkey}"):
                                    st.session_state.polygons[gkey] = scale_polygon(poly, 0.9)
                                    st.rerun()
        
        st.markdown("---")
        
        # Actions globales
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("💾 Sauvegarder (apprentissage)", type="primary", use_container_width=True):
                n_saved = 0
                for gname in polygons:
                    curr, orig = polygons.get(gname), st.session_state.original_polygons.get(gname)
                    if curr and orig and list(curr) != list(orig):
                        update_learned_params(gname, orig, curr)
                        n_saved += 1
                if n_saved:
                    st.success(f"✅ {n_saved} correction(s) sauvegardée(s)!")
                else:
                    st.info("Aucune modification")
        
        with col_b:
            if st.button("🔃 Réinitialiser tout", use_container_width=True):
                st.session_state.polygons = {k: list(v) if v else None for k, v in st.session_state.original_polygons.items()}
                st.session_state.selected_gate = None
                st.session_state.selected_point_idx = None
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
        c1.download_button("📥 CSV", df.to_csv(index=False), f"{reader.filename}.csv", "text/csv", use_container_width=True)
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        buf.seek(0)
        c2.download_button("📥 Excel", buf, f"{reader.filename}.xlsx", use_container_width=True)

else:
    st.markdown("""
    <div class="info-box">
    <h3>🔬 Gates Hexagonaux Interactifs</h3>
    <p><b>Fonctionnalités:</b></p>
    <ul>
    <li>Gates en <b>hexagones</b> (6 sommets)</li>
    <li>Modification <b>directe sur le graphique</b> par clic</li>
    <li>Mise à jour <b>en cascade</b> de toutes les populations</li>
    <li>Apprentissage automatique des corrections</li>
    </ul>
    <p>Uploadez un fichier FCS pour commencer.</p>
    </div>
    """, unsafe_allow_html=True)

st.caption(f"🔬 FACS Hexagones Interactifs | 🧠 {n_learned} corrections apprises")
