#!/usr/bin/env python3
"""
FACS Autogating - Gates Polygonaux Interactifs v3
- Gates polygonaux (convex hull) au lieu de rectangles
- Modification directe des sommets sur le graphique
- Sélection d'un point puis déplacement
- Auto-gating GMM + apprentissage
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull
from pathlib import Path
import tempfile
import io
import json
import os
from datetime import datetime
import flowio

st.set_page_config(page_title="FACS - Polygones Interactifs", page_icon="🔬", layout="wide")

LEARNED_PARAMS_FILE = "learned_gating_params.json"

st.markdown("""
<style>
.main-header { font-size: 1.8rem; color: #2c3e50; text-align: center; margin-bottom: 0.5rem; }
.info-box { background: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid #0066cc; margin: 0.5rem 0; }
.edit-panel { background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border: 1px solid #dee2e6; }
.point-btn { margin: 2px; }
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


def auto_gate_gmm(data, x_ch, y_ch, parent_mask=None, n_comp=2, mode='main'):
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
        # Convex Hull
        pts = np.column_stack([cx, cy])
        hull = ConvexHull(pts)
        hp = pts[hull.vertices]
        center = hp.mean(axis=0)
        # Expand 10%
        polygon = [(center[0] + 1.1 * (p[0] - center[0]), center[1] + 1.1 * (p[1] - center[1])) for p in hp]
        # Simplifier à 8-12 points max
        if len(polygon) > 12:
            step = max(1, len(polygon) // 10)
            polygon = polygon[::step]
        return polygon
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


def create_interactive_plot(data, x_ch, y_ch, x_label, y_label, title, polygon, parent_mask, gate_name):
    """Crée un graphique Plotly avec polygone et points de contrôle"""
    
    if x_ch is None or y_ch is None:
        fig = go.Figure()
        fig.add_annotation(text=f"Canal non trouvé", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
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
    if len(xt) > 8000:
        idx = np.random.choice(len(xt), 8000, replace=False)
        xd, yd = xt[idx], yt[idx]
    else:
        xd, yd = xt, yt
    
    fig = go.Figure()
    
    # Scatter plot
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
        
        # Polygone - remplissage
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
        
        # Points de contrôle (sommets)
        fig.add_trace(go.Scatter(
            x=[p[0] for p in polygon],
            y=[p[1] for p in polygon],
            mode='markers+text',
            marker=dict(size=14, color='red', symbol='circle',
                       line=dict(color='darkred', width=2)),
            text=[str(i+1) for i in range(len(polygon))],
            textposition='top center',
            textfont=dict(size=11, color='darkred', family='Arial Black'),
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
            font=dict(size=11, color='darkred'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='red',
            borderwidth=1
        )
    
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sup>Parent: {n_parent:,}</sup>", x=0.5, font=dict(size=12)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False,
        height=420,
        margin=dict(l=50, r=20, t=60, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False, showline=True, linecolor='#ccc'),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False, showline=True, linecolor='#ccc'),
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


def move_point(polygon, point_idx, new_x, new_y):
    """Déplace un point spécifique du polygone"""
    if polygon is None or point_idx >= len(polygon):
        return polygon
    new_poly = list(polygon)
    new_poly[point_idx] = (new_x, new_y)
    return new_poly


# ===== MAIN =====
st.markdown('<h1 class="main-header">🔬 FACS - Gates Polygonaux Interactifs</h1>', unsafe_allow_html=True)

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

# Pour l'édition des points
if 'edit_gate' not in st.session_state:
    st.session_state.edit_gate = None
if 'edit_point' not in st.session_state:
    st.session_state.edit_point = None

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
            poly = auto_gate_gmm(data, ch['FSC-A'], ch['SSC-A'], None, 2, 'main')
            poly = apply_learned_adj(poly, 'cells')
            st.session_state.polygons['cells'] = poly
            st.session_state.original_polygons['cells'] = list(poly) if poly else None
            prog.progress(25)
            
            # Singlets
            if ch['FSC-H']:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], poly, None)
                poly = auto_gate_gmm(data, ch['FSC-A'], ch['FSC-H'], cells_m, 2, 'main')
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
                poly = auto_gate_gmm(data, ch['LiveDead'], ch['SSC-A'], sing_m, 2, 'low_x')
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
                poly = auto_gate_gmm(data, ch['hCD45'], ch['SSC-A'], live_m, 2, 'high_x')
                poly = apply_learned_adj(poly, 'hcd45')
            else:
                poly = None
            st.session_state.polygons['hcd45'] = poly
            st.session_state.original_polygons['hcd45'] = list(poly) if poly else None
            prog.progress(100)
            
            st.session_state.auto_done = True
            st.rerun()
    
    # AFFICHAGE
    if st.session_state.auto_done:
        polygons = st.session_state.polygons
        
        # Recalcul masques
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
        
        # Affichage en grille 2x2
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
                    st.markdown(f"### {gate_idx + 1}️⃣ {gname}")
                    
                    # Graphique
                    fig, n_in, pct = create_interactive_plot(
                        data, x_ch, y_ch, x_label, y_label, title,
                        polygons.get(gkey), parent_mask, gname
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{gkey}")
                    
                    stats.append((gname, title.split('→')[0].strip() if '→' in title else 'Ungated', n_in, pct))
                    
                    # Panel d'édition
                    if polygons.get(gkey):
                        with st.expander(f"✏️ Modifier {gname}", expanded=False):
                            poly = polygons[gkey]
                            n_points = len(poly)
                            
                            # Sélection du point
                            col_a, col_b = st.columns([1, 2])
                            with col_a:
                                point_idx = st.selectbox(
                                    "Point",
                                    range(n_points),
                                    format_func=lambda x: f"Point {x+1}",
                                    key=f"pt_sel_{gkey}"
                                )
                            
                            with col_b:
                                st.caption(f"Actuel: ({poly[point_idx][0]:.1f}, {poly[point_idx][1]:.1f})")
                            
                            # Nouvelles coordonnées
                            col_x, col_y = st.columns(2)
                            with col_x:
                                new_x = st.number_input("X", value=float(poly[point_idx][0]), 
                                                       step=1.0, key=f"nx_{gkey}")
                            with col_y:
                                new_y = st.number_input("Y", value=float(poly[point_idx][1]), 
                                                       step=1.0, key=f"ny_{gkey}")
                            
                            col_btn1, col_btn2 = st.columns(2)
                            with col_btn1:
                                if st.button("✅ Appliquer", key=f"apply_{gkey}", use_container_width=True):
                                    st.session_state.polygons[gkey] = move_point(poly, point_idx, new_x, new_y)
                                    st.rerun()
                            with col_btn2:
                                if st.button("🔄 Reset point", key=f"reset_pt_{gkey}", use_container_width=True):
                                    orig = st.session_state.original_polygons.get(gkey)
                                    if orig and point_idx < len(orig):
                                        st.session_state.polygons[gkey] = move_point(
                                            poly, point_idx, orig[point_idx][0], orig[point_idx][1]
                                        )
                                        st.rerun()
                            
                            st.markdown("---")
                            st.markdown("**Déplacer tout le gate:**")
                            
                            move_step = st.slider("Pas", 1, 20, 5, key=f"step_{gkey}")
                            
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            with col_m1:
                                if st.button("⬆️", key=f"up_{gkey}", use_container_width=True):
                                    st.session_state.polygons[gkey] = move_polygon(poly, 0, move_step)
                                    st.rerun()
                            with col_m2:
                                if st.button("⬇️", key=f"dn_{gkey}", use_container_width=True):
                                    st.session_state.polygons[gkey] = move_polygon(poly, 0, -move_step)
                                    st.rerun()
                            with col_m3:
                                if st.button("⬅️", key=f"lt_{gkey}", use_container_width=True):
                                    st.session_state.polygons[gkey] = move_polygon(poly, -move_step, 0)
                                    st.rerun()
                            with col_m4:
                                if st.button("➡️", key=f"rt_{gkey}", use_container_width=True):
                                    st.session_state.polygons[gkey] = move_polygon(poly, move_step, 0)
                                    st.rerun()
                            
                            col_s1, col_s2 = st.columns(2)
                            with col_s1:
                                if st.button("➕ Agrandir", key=f"grow_{gkey}", use_container_width=True):
                                    st.session_state.polygons[gkey] = scale_polygon(poly, 1.1)
                                    st.rerun()
                            with col_s2:
                                if st.button("➖ Réduire", key=f"shrink_{gkey}", use_container_width=True):
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
                st.rerun()
        
        with col_c:
            if st.button("🔄 Rafraîchir", use_container_width=True):
                st.rerun()
        
        # Résumé
        st.markdown("### 📊 Résumé")
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
    <h3>🔬 Gates Polygonaux Interactifs</h3>
    <p><b>Nouveautés v3:</b></p>
    <ul>
    <li>Gates en <b>polygones</b> (convex hull) au lieu de rectangles</li>
    <li>Modification <b>point par point</b> des sommets</li>
    <li>Déplacement et redimensionnement global</li>
    <li>Apprentissage automatique des corrections</li>
    </ul>
    <p>Uploadez un fichier FCS pour commencer.</p>
    </div>
    """, unsafe_allow_html=True)

st.caption(f"🔬 FACS Polygones Interactifs v3 | 🧠 {n_learned} corrections apprises")
