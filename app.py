#!/usr/bin/env python3
"""
FACS Autogating - Modification des Gates par Points de Contrôle
- Auto-gating automatique (GMM)  
- Points draggables sur les graphiques pour modifier les gates
- Recalcul automatique des populations enfants
- Apprentissage des corrections
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull
from pathlib import Path as FilePath
import tempfile
import io
import json
import os
from datetime import datetime
import flowio
import re

# Configuration
st.set_page_config(
    page_title="FACS - Auto-Gating",
    page_icon="🔬",
    layout="wide"
)

LEARNED_PARAMS_FILE = "learned_gating_params.json"

st.markdown("""
    <style>
    .main-header { font-size: 1.8rem; color: #2c3e50; text-align: center; margin-bottom: 0.5rem; }
    .info-box { background: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem;
                border-left: 4px solid #0066cc; margin: 0.5rem 0; }
    .learning-box { background: #f0e6ff; padding: 0.6rem; border-radius: 0.5rem;
                    border-left: 4px solid #6f42c1; margin: 0.3rem 0; font-size: 0.9rem; }
    .gate-edit { background: #fff8e7; padding: 0.6rem; border-radius: 0.4rem;
                 border: 1px solid #ffc107; margin: 0.3rem 0; }
    </style>
""", unsafe_allow_html=True)


def load_learned_params():
    if os.path.exists(LEARNED_PARAMS_FILE):
        try:
            with open(LEARNED_PARAMS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'n_corrections': 0, 'gates': {}, 'last_updated': None}


def save_learned_params(params):
    params['last_updated'] = datetime.now().isoformat()
    try:
        with open(LEARNED_PARAMS_FILE, 'w') as f:
            json.dump(params, f, indent=2)
        return True
    except:
        return False


def update_learned_params(gate_name, original_polygon, corrected_polygon):
    params = load_learned_params()
    
    if gate_name not in params['gates']:
        params['gates'][gate_name] = {'avg_adjustment': {'x': 0, 'y': 0}, 'n_samples': 0}
    
    gate_params = params['gates'][gate_name]
    
    orig_center = np.mean(original_polygon, axis=0)
    corr_center = np.mean(corrected_polygon, axis=0)
    
    dx = float(corr_center[0] - orig_center[0])
    dy = float(corr_center[1] - orig_center[1])
    
    gate_params['n_samples'] += 1
    n = gate_params['n_samples']
    alpha = 2 / (n + 1)
    
    gate_params['avg_adjustment']['x'] = (1 - alpha) * gate_params['avg_adjustment']['x'] + alpha * dx
    gate_params['avg_adjustment']['y'] = (1 - alpha) * gate_params['avg_adjustment']['y'] + alpha * dy
    
    params['n_corrections'] += 1
    save_learned_params(params)
    return params


class FCSReader:
    def __init__(self, fcs_path):
        self.fcs_path = fcs_path
        self.flow_data = flowio.FlowData(fcs_path)
        self.filename = FilePath(fcs_path).stem
        self.data = None
        self.channels = []
        self.channel_markers = {}
        self._load()
    
    def _load(self):
        events = np.array(self.flow_data.events, dtype=np.float64)
        n_ch = self.flow_data.channel_count
        
        if events.ndim == 1:
            events = events.reshape(-1, n_ch)
        
        labels = []
        for i in range(1, n_ch + 1):
            pnn = self.flow_data.text.get(f'$P{i}N', f'Ch{i}').strip()
            pns = self.flow_data.text.get(f'$P{i}S', '').strip()
            
            marker = pnn
            if pns:
                m = re.search(r'[hm]?(CD\d+[a-z]?|FoxP3|Granzyme|PD[L]?1|HLA|Viab)', pns, re.I)
                if m:
                    marker = m.group(0).lstrip('hm')
            
            self.channel_markers[pnn] = marker
            labels.append(pnn)
        
        self.channels = labels
        self.data = pd.DataFrame(events, columns=labels)


def find_channel(data, keywords):
    for col in data.columns:
        for kw in keywords:
            if kw.upper() in col.upper():
                return col
    return None


def biex(x):
    return np.arcsinh(np.asarray(x, float) / 150) * 50


def point_in_polygon(x, y, polygon):
    """Vérifie si les points sont dans le polygone"""
    if polygon is None or len(polygon) < 3:
        return np.zeros(len(x), dtype=bool)
    
    path = Path(polygon)
    points = np.column_stack([x, y])
    return path.contains_points(points)


def apply_gate(data, x_ch, y_ch, polygon, parent_mask=None):
    """Applique un gate et retourne le masque"""
    if polygon is None or len(polygon) < 3:
        return pd.Series(False, index=data.index)
    
    if parent_mask is not None:
        base = parent_mask.values
    else:
        base = np.ones(len(data), dtype=bool)
    
    x = data[x_ch].values
    y = data[y_ch].values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & base
    
    if not valid.any():
        return pd.Series(False, index=data.index)
    
    xt = biex(x)
    yt = biex(y)
    
    in_poly = point_in_polygon(xt, yt, polygon)
    
    result = np.zeros(len(data), dtype=bool)
    result[valid & in_poly] = True
    return pd.Series(result, index=data.index)


def auto_gate_gmm(data, x_ch, y_ch, parent_mask=None, n_comp=2, mode='main'):
    """Auto-gating avec GMM"""
    if parent_mask is not None:
        subset = data[parent_mask]
    else:
        subset = data
    
    x = subset[x_ch].values
    y = subset[y_ch].values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    
    if valid.sum() < 100:
        return None
    
    xt = biex(x[valid])
    yt = biex(y[valid])
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
            scores = [np.mean(xt[labels == i]) - np.mean(yt[labels == i]) for i in range(n_comp)]
            target = np.argmax(scores)
        else:
            target = 0
        
        mask = labels == target
        cx, cy = xt[mask], yt[mask]
        
        if len(cx) < 30:
            return None
        
        pts = np.column_stack([cx, cy])
        hull = ConvexHull(pts)
        hp = pts[hull.vertices]
        
        center = hp.mean(axis=0)
        polygon = [(center[0] + 1.1 * (p[0] - center[0]), 
                    center[1] + 1.1 * (p[1] - center[1])) for p in hp]
        
        if len(polygon) > 12:
            polygon = polygon[::max(1, len(polygon)//10)]
        
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


def create_gate_plot(data, x_ch, y_ch, x_label, y_label, title, polygon, parent_mask, gate_name, show_vertices=True):
    """Crée un graphique matplotlib avec le gate"""
    
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=100)
    
    if parent_mask is not None:
        subset = data[parent_mask]
    else:
        subset = data
    
    n_parent = len(subset)
    
    if n_parent == 0:
        ax.text(0.5, 0.5, "Pas de données", ha='center', va='center', transform=ax.transAxes)
        return fig, 0, 0
    
    x = subset[x_ch].values
    y = subset[y_ch].values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    
    xt = biex(x[valid])
    yt = biex(y[valid])
    
    # Sous-échantillonner
    if len(xt) > 8000:
        idx = np.random.choice(len(xt), 8000, replace=False)
        xd, yd = xt[idx], yt[idx]
    else:
        xd, yd = xt, yt
    
    # Scatter plot
    ax.scatter(xd, yd, s=1, c=yd, cmap='viridis', alpha=0.5, rasterized=True)
    
    # Stats du gate
    n_in = 0
    pct = 0
    
    if polygon and len(polygon) >= 3:
        # Dessiner le polygone
        poly_patch = patches.Polygon(polygon, fill=True, facecolor='red', 
                                      edgecolor='red', alpha=0.15, linewidth=2)
        ax.add_patch(poly_patch)
        
        # Contour
        poly_x = [p[0] for p in polygon] + [polygon[0][0]]
        poly_y = [p[1] for p in polygon] + [polygon[0][1]]
        ax.plot(poly_x, poly_y, 'r-', linewidth=2)
        
        # Sommets (points de contrôle)
        if show_vertices:
            for i, (px, py) in enumerate(polygon):
                ax.plot(px, py, 'ro', markersize=8, markeredgecolor='darkred', markeredgewidth=1.5)
                ax.annotate(str(i+1), (px, py), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='darkred', fontweight='bold')
        
        # Calculer stats
        full_mask = apply_gate(data, x_ch, y_ch, polygon, parent_mask)
        n_in = full_mask.sum()
        pct = n_in / n_parent * 100 if n_parent > 0 else 0
        
        # Annotation au centre
        cx = np.mean([p[0] for p in polygon])
        cy = np.mean([p[1] for p in polygon])
        ax.annotate(f"{gate_name}\n{pct:.1f}%\n({n_in:,})", (cx, cy),
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   color='darkred', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title(f"{title}\n(n={n_parent:,})", fontsize=10, fontweight='bold')
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig, n_in, pct


def modify_polygon_vertex(polygon, vertex_index, dx, dy):
    """Modifie un sommet du polygone"""
    if polygon is None or vertex_index >= len(polygon):
        return polygon
    
    new_poly = [list(p) for p in polygon]
    new_poly[vertex_index][0] += dx
    new_poly[vertex_index][1] += dy
    return [tuple(p) for p in new_poly]


def move_polygon(polygon, dx, dy):
    """Déplace tout le polygone"""
    if polygon is None:
        return None
    return [(p[0] + dx, p[1] + dy) for p in polygon]


def scale_polygon(polygon, factor):
    """Agrandit/réduit le polygone"""
    if polygon is None:
        return None
    center = np.mean(polygon, axis=0)
    return [(center[0] + factor * (p[0] - center[0]), 
             center[1] + factor * (p[1] - center[1])) for p in polygon]


# ==================== MAIN ====================

st.markdown('<h1 class="main-header">🔬 FACS Auto-Gating</h1>', unsafe_allow_html=True)

learned = load_learned_params()
n_learned = learned.get('n_corrections', 0)

if n_learned > 0:
    st.markdown(f'<div class="learning-box">🧠 <b>{n_learned}</b> correction(s) apprises - L\'algorithme s\'améliore !</div>', unsafe_allow_html=True)

# Session state
for key in ['data', 'reader', 'channels', 'polygons', 'original_polygons']:
    if key not in st.session_state:
        st.session_state[key] = {} if 'polygons' in key or key == 'channels' else None

if 'auto_done' not in st.session_state:
    st.session_state.auto_done = False

# Upload
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
            
            data = reader.data
            st.session_state.channels = {
                'FSC-A': find_channel(data, ['FSC-A']),
                'FSC-H': find_channel(data, ['FSC-H']),
                'SSC-A': find_channel(data, ['SSC-A']),
                'LiveDead': find_channel(data, ['LiveDead', 'Viab', 'Aqua']),
                'hCD45': find_channel(data, ['PerCP']),
                'CD3': find_channel(data, ['AF488']),
                'CD19': find_channel(data, ['PE-Fire700']),
                'CD4': find_channel(data, ['BV650']),
                'CD8': find_channel(data, ['BUV805']),
            }
    
    reader = st.session_state.reader
    data = st.session_state.data
    ch = st.session_state.channels
    n_total = len(data)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Événements", f"{n_total:,}")
    c2.metric("Canaux", len(reader.channels))
    c3.metric("Fichier", reader.filename[:25])
    
    st.markdown("---")
    
    # ==================== AUTO-GATING ====================
    
    if not st.session_state.auto_done:
        if st.button("🚀 **LANCER L'AUTO-GATING**", type="primary", use_container_width=True):
            prog = st.progress(0)
            
            # Cells
            poly = auto_gate_gmm(data, ch['FSC-A'], ch['SSC-A'], None, 2, 'main')
            poly = apply_learned_adj(poly, 'cells')
            st.session_state.polygons['cells'] = poly
            st.session_state.original_polygons['cells'] = list(poly) if poly else None
            prog.progress(15)
            
            # Singlets
            cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], poly, None)
            poly = auto_gate_gmm(data, ch['FSC-A'], ch['FSC-H'], cells_m, 2, 'main')
            poly = apply_learned_adj(poly, 'singlets')
            st.session_state.polygons['singlets'] = poly
            st.session_state.original_polygons['singlets'] = list(poly) if poly else None
            prog.progress(30)
            
            # Live
            sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], poly, cells_m)
            poly = auto_gate_gmm(data, ch['LiveDead'], ch['SSC-A'], sing_m, 2, 'low_x')
            poly = apply_learned_adj(poly, 'live')
            st.session_state.polygons['live'] = poly
            st.session_state.original_polygons['live'] = list(poly) if poly else None
            prog.progress(50)
            
            # hCD45
            live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'], poly, sing_m)
            poly = auto_gate_gmm(data, ch['hCD45'], ch['SSC-A'], live_m, 2, 'high_x')
            poly = apply_learned_adj(poly, 'hcd45')
            st.session_state.polygons['hcd45'] = poly
            st.session_state.original_polygons['hcd45'] = list(poly) if poly else None
            prog.progress(70)
            
            # T cells
            hcd45_m = apply_gate(data, ch['hCD45'], ch['SSC-A'], poly, live_m)
            poly = auto_gate_gmm(data, ch['CD3'], ch['CD19'], hcd45_m, 3, 'high_x_low_y')
            poly = apply_learned_adj(poly, 't_cells')
            st.session_state.polygons['t_cells'] = poly
            st.session_state.original_polygons['t_cells'] = list(poly) if poly else None
            prog.progress(85)
            
            # CD4+
            t_m = apply_gate(data, ch['CD3'], ch['CD19'], poly, hcd45_m)
            poly = auto_gate_gmm(data, ch['CD4'], ch['CD8'], t_m, 3, 'high_x_low_y')
            poly = apply_learned_adj(poly, 'cd4')
            st.session_state.polygons['cd4'] = poly
            st.session_state.original_polygons['cd4'] = list(poly) if poly else None
            prog.progress(100)
            
            st.session_state.auto_done = True
            st.rerun()
    
    # ==================== AFFICHAGE & ÉDITION ====================
    
    if st.session_state.auto_done:
        polygons = st.session_state.polygons
        
        st.markdown("""
        <div class="info-box">
        <b>📌 Modification des gates :</b> Utilisez les contrôles sous chaque graphique pour ajuster les polygones.<br>
        Les <b>points rouges numérotés</b> sont les sommets du gate que vous pouvez déplacer.
        </div>
        """, unsafe_allow_html=True)
        
        # Calculer tous les masques avec les polygones actuels
        cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], polygons.get('cells'), None)
        sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], polygons.get('singlets'), cells_m)
        live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'], polygons.get('live'), sing_m)
        hcd45_m = apply_gate(data, ch['hCD45'], ch['SSC-A'], polygons.get('hcd45'), live_m)
        t_m = apply_gate(data, ch['CD3'], ch['CD19'], polygons.get('t_cells'), hcd45_m)
        cd4_m = apply_gate(data, ch['CD4'], ch['CD8'], polygons.get('cd4'), t_m)
        
        # GATE 1: CELLS
        st.markdown("### 1️⃣ Cells")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, n_cells, pct_cells = create_gate_plot(
                data, ch['FSC-A'], ch['SSC-A'], 'FSC-A', 'SSC-A',
                'Ungated → Cells', polygons.get('cells'), None, 'Cells'
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        with col2:
            st.markdown('<div class="gate-edit">', unsafe_allow_html=True)
            st.markdown(f"**Cells:** {n_cells:,} ({pct_cells:.1f}%)")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⬆️", key="cells_up"):
                    st.session_state.polygons['cells'] = move_polygon(polygons['cells'], 0, 5)
                    st.rerun()
                if st.button("⬇️", key="cells_down"):
                    st.session_state.polygons['cells'] = move_polygon(polygons['cells'], 0, -5)
                    st.rerun()
            with col_b:
                if st.button("⬅️", key="cells_left"):
                    st.session_state.polygons['cells'] = move_polygon(polygons['cells'], -5, 0)
                    st.rerun()
                if st.button("➡️", key="cells_right"):
                    st.session_state.polygons['cells'] = move_polygon(polygons['cells'], 5, 0)
                    st.rerun()
            
            col_c, col_d = st.columns(2)
            with col_c:
                if st.button("➕ Agrandir", key="cells_grow"):
                    st.session_state.polygons['cells'] = scale_polygon(polygons['cells'], 1.1)
                    st.rerun()
            with col_d:
                if st.button("➖ Réduire", key="cells_shrink"):
                    st.session_state.polygons['cells'] = scale_polygon(polygons['cells'], 0.9)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # GATE 2: SINGLETS
        st.markdown("### 2️⃣ Singlets")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, n_sing, pct_sing = create_gate_plot(
                data, ch['FSC-A'], ch['FSC-H'], 'FSC-A', 'FSC-H',
                'Cells → Singlets', polygons.get('singlets'), cells_m, 'Singlets'
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        with col2:
            st.markdown('<div class="gate-edit">', unsafe_allow_html=True)
            st.markdown(f"**Singlets:** {n_sing:,} ({pct_sing:.1f}%)")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⬆️", key="sing_up"):
                    st.session_state.polygons['singlets'] = move_polygon(polygons['singlets'], 0, 5)
                    st.rerun()
                if st.button("⬇️", key="sing_down"):
                    st.session_state.polygons['singlets'] = move_polygon(polygons['singlets'], 0, -5)
                    st.rerun()
            with col_b:
                if st.button("⬅️", key="sing_left"):
                    st.session_state.polygons['singlets'] = move_polygon(polygons['singlets'], -5, 0)
                    st.rerun()
                if st.button("➡️", key="sing_right"):
                    st.session_state.polygons['singlets'] = move_polygon(polygons['singlets'], 5, 0)
                    st.rerun()
            
            col_c, col_d = st.columns(2)
            with col_c:
                if st.button("➕ Agrandir", key="sing_grow"):
                    st.session_state.polygons['singlets'] = scale_polygon(polygons['singlets'], 1.1)
                    st.rerun()
            with col_d:
                if st.button("➖ Réduire", key="sing_shrink"):
                    st.session_state.polygons['singlets'] = scale_polygon(polygons['singlets'], 0.9)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # GATE 3: LIVE
        st.markdown("### 3️⃣ Live")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, n_live, pct_live = create_gate_plot(
                data, ch['LiveDead'], ch['SSC-A'], 'Live/Dead', 'SSC-A',
                'Singlets → Live', polygons.get('live'), sing_m, 'Live'
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        with col2:
            st.markdown('<div class="gate-edit">', unsafe_allow_html=True)
            st.markdown(f"**Live:** {n_live:,} ({pct_live:.1f}%)")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⬆️", key="live_up"):
                    st.session_state.polygons['live'] = move_polygon(polygons['live'], 0, 5)
                    st.rerun()
                if st.button("⬇️", key="live_down"):
                    st.session_state.polygons['live'] = move_polygon(polygons['live'], 0, -5)
                    st.rerun()
            with col_b:
                if st.button("⬅️", key="live_left"):
                    st.session_state.polygons['live'] = move_polygon(polygons['live'], -5, 0)
                    st.rerun()
                if st.button("➡️", key="live_right"):
                    st.session_state.polygons['live'] = move_polygon(polygons['live'], 5, 0)
                    st.rerun()
            
            col_c, col_d = st.columns(2)
            with col_c:
                if st.button("➕ Agrandir", key="live_grow"):
                    st.session_state.polygons['live'] = scale_polygon(polygons['live'], 1.1)
                    st.rerun()
            with col_d:
                if st.button("➖ Réduire", key="live_shrink"):
                    st.session_state.polygons['live'] = scale_polygon(polygons['live'], 0.9)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # GATE 4: hCD45
        st.markdown("### 4️⃣ hCD45+")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, n_hcd45, pct_hcd45 = create_gate_plot(
                data, ch['hCD45'], ch['SSC-A'], 'hCD45', 'SSC-A',
                'Live → hCD45+', polygons.get('hcd45'), live_m, 'hCD45+'
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        with col2:
            st.markdown('<div class="gate-edit">', unsafe_allow_html=True)
            st.markdown(f"**hCD45+:** {n_hcd45:,} ({pct_hcd45:.1f}%)")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⬆️", key="hcd45_up"):
                    st.session_state.polygons['hcd45'] = move_polygon(polygons['hcd45'], 0, 5)
                    st.rerun()
                if st.button("⬇️", key="hcd45_down"):
                    st.session_state.polygons['hcd45'] = move_polygon(polygons['hcd45'], 0, -5)
                    st.rerun()
            with col_b:
                if st.button("⬅️", key="hcd45_left"):
                    st.session_state.polygons['hcd45'] = move_polygon(polygons['hcd45'], -5, 0)
                    st.rerun()
                if st.button("➡️", key="hcd45_right"):
                    st.session_state.polygons['hcd45'] = move_polygon(polygons['hcd45'], 5, 0)
                    st.rerun()
            
            col_c, col_d = st.columns(2)
            with col_c:
                if st.button("➕ Agrandir", key="hcd45_grow"):
                    st.session_state.polygons['hcd45'] = scale_polygon(polygons['hcd45'], 1.1)
                    st.rerun()
            with col_d:
                if st.button("➖ Réduire", key="hcd45_shrink"):
                    st.session_state.polygons['hcd45'] = scale_polygon(polygons['hcd45'], 0.9)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # GATE 5: T CELLS
        st.markdown("### 5️⃣ T cells")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, n_tcells, pct_tcells = create_gate_plot(
                data, ch['CD3'], ch['CD19'], 'CD3', 'CD19',
                'hCD45+ → T cells', polygons.get('t_cells'), hcd45_m, 'T cells'
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        with col2:
            st.markdown('<div class="gate-edit">', unsafe_allow_html=True)
            st.markdown(f"**T cells:** {n_tcells:,} ({pct_tcells:.1f}%)")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⬆️", key="tcells_up"):
                    st.session_state.polygons['t_cells'] = move_polygon(polygons['t_cells'], 0, 5)
                    st.rerun()
                if st.button("⬇️", key="tcells_down"):
                    st.session_state.polygons['t_cells'] = move_polygon(polygons['t_cells'], 0, -5)
                    st.rerun()
            with col_b:
                if st.button("⬅️", key="tcells_left"):
                    st.session_state.polygons['t_cells'] = move_polygon(polygons['t_cells'], -5, 0)
                    st.rerun()
                if st.button("➡️", key="tcells_right"):
                    st.session_state.polygons['t_cells'] = move_polygon(polygons['t_cells'], 5, 0)
                    st.rerun()
            
            col_c, col_d = st.columns(2)
            with col_c:
                if st.button("➕ Agrandir", key="tcells_grow"):
                    st.session_state.polygons['t_cells'] = scale_polygon(polygons['t_cells'], 1.1)
                    st.rerun()
            with col_d:
                if st.button("➖ Réduire", key="tcells_shrink"):
                    st.session_state.polygons['t_cells'] = scale_polygon(polygons['t_cells'], 0.9)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # GATE 6: CD4+
        st.markdown("### 6️⃣ CD4+ T cells")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if t_m.sum() > 0:
                fig, n_cd4, pct_cd4 = create_gate_plot(
                    data, ch['CD4'], ch['CD8'], 'CD4', 'CD8',
                    'T cells → CD4+', polygons.get('cd4'), t_m, 'CD4+'
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("Pas de T cells détectés")
                n_cd4, pct_cd4 = 0, 0
        
        with col2:
            if t_m.sum() > 0:
                st.markdown('<div class="gate-edit">', unsafe_allow_html=True)
                st.markdown(f"**CD4+:** {n_cd4:,} ({pct_cd4:.1f}%)")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("⬆️", key="cd4_up"):
                        st.session_state.polygons['cd4'] = move_polygon(polygons['cd4'], 0, 5)
                        st.rerun()
                    if st.button("⬇️", key="cd4_down"):
                        st.session_state.polygons['cd4'] = move_polygon(polygons['cd4'], 0, -5)
                        st.rerun()
                with col_b:
                    if st.button("⬅️", key="cd4_left"):
                        st.session_state.polygons['cd4'] = move_polygon(polygons['cd4'], -5, 0)
                        st.rerun()
                    if st.button("➡️", key="cd4_right"):
                        st.session_state.polygons['cd4'] = move_polygon(polygons['cd4'], 5, 0)
                        st.rerun()
                
                col_c, col_d = st.columns(2)
                with col_c:
                    if st.button("➕ Agrandir", key="cd4_grow"):
                        st.session_state.polygons['cd4'] = scale_polygon(polygons['cd4'], 1.1)
                        st.rerun()
                with col_d:
                    if st.button("➖ Réduire", key="cd4_shrink"):
                        st.session_state.polygons['cd4'] = scale_polygon(polygons['cd4'], 0.9)
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ACTIONS GLOBALES
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("🔄 Rafraîchir", use_container_width=True):
                st.rerun()
        
        with col_b:
            if st.button("💾 **Sauvegarder (apprentissage)**", type="primary", use_container_width=True):
                n_saved = 0
                for gname in polygons:
                    curr = polygons.get(gname)
                    orig = st.session_state.original_polygons.get(gname)
                    if curr and orig and list(curr) != list(orig):
                        update_learned_params(gname, orig, curr)
                        n_saved += 1
                if n_saved:
                    st.success(f"✅ {n_saved} correction(s) sauvegardée(s) ! L'algorithme s'améliorera.")
                else:
                    st.info("Aucune modification détectée")
        
        with col_c:
            if st.button("🔃 Réinitialiser", use_container_width=True):
                st.session_state.polygons = {k: list(v) if v else None 
                                             for k, v in st.session_state.original_polygons.items()}
                st.rerun()
        
        # TABLEAU RÉCAPITULATIF
        st.markdown("### 📊 Résumé des Populations")
        
        stats_data = [
            ('Cells', 'Ungated', n_cells, pct_cells),
            ('Singlets', 'Cells', n_sing, pct_sing),
            ('Live', 'Singlets', n_live, pct_live),
            ('hCD45+', 'Live', n_hcd45, pct_hcd45),
            ('T cells', 'hCD45+', n_tcells, pct_tcells),
            ('CD4+ T', 'T cells', n_cd4, pct_cd4),
        ]
        
        df = pd.DataFrame(stats_data, columns=['Population', 'Parent', 'Count', '% Parent'])
        df['% Total'] = (df['Count'] / n_total * 100).round(2)
        df['% Parent'] = df['% Parent'].round(1)
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 CSV", df.to_csv(index=False), 
                              f"{reader.filename}_stats.csv", "text/csv", use_container_width=True)
        with c2:
            buf = io.BytesIO()
            df.to_excel(buf, index=False, engine='openpyxl')
            buf.seek(0)
            st.download_button("📥 Excel", buf, f"{reader.filename}_stats.xlsx", use_container_width=True)

else:
    st.markdown("""
    <div class="info-box">
    <h3>🔬 Auto-Gating Intelligent</h3>
    <ol>
    <li>Uploadez un fichier FCS</li>
    <li>Lancez l'auto-gating</li>
    <li>Ajustez les gates avec les boutons (déplacer, agrandir, réduire)</li>
    <li>Sauvegardez vos corrections pour améliorer l'algorithme</li>
    </ol>
    <p>Plus vous corrigez, plus l'auto-gating devient précis ! 🧠</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption(f"🔬 FACS Auto-Gating | 🧠 {n_learned} correction(s) apprises")
