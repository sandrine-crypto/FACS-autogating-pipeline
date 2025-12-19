#!/usr/bin/env python3
"""
FACS Autogating - Auto-Gating Intelligent avec Apprentissage
- Auto-gating automatique optimisé (GMM + heuristiques)
- Correction manuelle des gates avec recalcul automatique des populations
- Apprentissage des corrections pour améliorer les futurs auto-gatings
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
import re

# Configuration
st.set_page_config(
    page_title="FACS - Auto-Gating Intelligent",
    page_icon="🔬",
    layout="wide"
)

# Fichier de sauvegarde des paramètres appris
LEARNED_PARAMS_FILE = "learned_gating_params.json"

# CSS
st.markdown("""
    <style>
    .main-header { font-size: 2rem; color: #2c3e50; text-align: center; margin-bottom: 1rem; }
    .success-box { background: #d4edda; padding: 0.8rem; border-radius: 0.5rem;
                   border-left: 4px solid #28a745; margin: 0.5rem 0; }
    .warning-box { background: #fff3cd; padding: 0.8rem; border-radius: 0.5rem;
                   border-left: 4px solid #ffc107; margin: 0.5rem 0; }
    .info-box { background: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem;
                border-left: 4px solid #0066cc; margin: 0.5rem 0; }
    .learning-box { background: #f0e6ff; padding: 0.8rem; border-radius: 0.5rem;
                    border-left: 4px solid #6f42c1; margin: 0.5rem 0; }
    .stats-card { background: white; padding: 0.5rem; border-radius: 0.3rem;
                  border: 1px solid #ddd; margin: 0.2rem 0; }
    </style>
""", unsafe_allow_html=True)


def load_learned_params():
    """Charge les paramètres appris depuis le fichier"""
    if os.path.exists(LEARNED_PARAMS_FILE):
        try:
            with open(LEARNED_PARAMS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'n_corrections': 0, 'gates': {}, 'last_updated': None}


def save_learned_params(params):
    """Sauvegarde les paramètres appris"""
    params['last_updated'] = datetime.now().isoformat()
    try:
        with open(LEARNED_PARAMS_FILE, 'w') as f:
            json.dump(params, f, indent=2)
        return True
    except:
        return False


def update_learned_params(gate_name, original_polygon, corrected_polygon, data_stats):
    """Met à jour les paramètres appris avec une correction"""
    params = load_learned_params()
    
    if gate_name not in params['gates']:
        params['gates'][gate_name] = {
            'corrections': [],
            'avg_adjustment': {'x': 0, 'y': 0},
            'n_samples': 0
        }
    
    gate_params = params['gates'][gate_name]
    
    orig_center = np.mean(original_polygon, axis=0)
    corr_center = np.mean(corrected_polygon, axis=0)
    
    adjustment = {
        'dx': float(corr_center[0] - orig_center[0]),
        'dy': float(corr_center[1] - orig_center[1]),
    }
    
    gate_params['corrections'].append(adjustment)
    gate_params['n_samples'] += 1
    
    n = gate_params['n_samples']
    alpha = 2 / (n + 1)
    
    gate_params['avg_adjustment']['x'] = (1 - alpha) * gate_params['avg_adjustment']['x'] + alpha * adjustment['dx']
    gate_params['avg_adjustment']['y'] = (1 - alpha) * gate_params['avg_adjustment']['y'] + alpha * adjustment['dy']
    
    params['n_corrections'] += 1
    save_learned_params(params)
    
    return params


class FCSReader:
    def __init__(self, fcs_path):
        self.fcs_path = fcs_path
        self.flow_data = flowio.FlowData(fcs_path)
        self.data = None
        self.channels = []
        self.channel_markers = {}
        self.filename = Path(fcs_path).stem
        self.load_data()
    
    def load_data(self):
        events = self.flow_data.events
        n_channels = self.flow_data.channel_count
        
        if not isinstance(events, np.ndarray):
            events = np.array(events, dtype=np.float64)
        
        if events.ndim == 1:
            n_events = len(events) // n_channels
            events = events.reshape(n_events, n_channels)
        
        pnn_labels = []
        for i in range(1, n_channels + 1):
            pnn = self.flow_data.text.get(f'$P{i}N', None) or self.flow_data.text.get(f'p{i}n', f'Ch{i}')
            pns = self.flow_data.text.get(f'$P{i}S', None) or self.flow_data.text.get(f'p{i}s', '')
            pnn = pnn.strip() if pnn else f'Ch{i}'
            pns = pns.strip() if pns else ''
            
            marker = pnn
            if pns:
                match = re.search(r'[hm]?(CD\d+[a-z]?|FoxP3|Granzyme|PD[L]?1|HLA|Viab)', pns, re.IGNORECASE)
                if match:
                    marker = match.group(0).lstrip('hm')
            
            self.channel_markers[pnn] = marker
            pnn_labels.append(pnn)
        
        self.channels = pnn_labels
        self.data = pd.DataFrame(events, columns=self.channels)
        return self.data


def find_channel(data, keywords):
    for col in data.columns:
        col_upper = col.upper()
        for kw in keywords:
            if kw.upper() in col_upper:
                return col
    return None


def biex_transform(x):
    x = np.asarray(x, dtype=float)
    return np.arcsinh(x / 150) * 50


def biex_inverse(y):
    return np.sinh(y / 50) * 150


def point_in_polygon(x, y, poly_x, poly_y):
    """Ray casting algorithm"""
    n = len(poly_x)
    inside = np.zeros(len(x), dtype=bool)
    
    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        
        cond1 = (yi > y) != (yj > y)
        slope = (xj - xi) / (yj - yi + 1e-10)
        x_intersect = xi + slope * (y - yi)
        cond2 = x < x_intersect
        
        inside = inside ^ (cond1 & cond2)
        j = i
    
    return inside


def parse_coordinates(coords_str):
    """Parse les coordonnées depuis une chaîne"""
    if not coords_str or not coords_str.strip():
        return None
    try:
        coords = []
        for p in coords_str.split(';'):
            if ',' in p:
                parts = p.strip().split(',')
                if len(parts) >= 2:
                    coords.append([float(parts[0]), float(parts[1])])
        return coords if len(coords) >= 3 else None
    except:
        return None


def format_coordinates(polygon):
    """Formate les coordonnées en chaîne"""
    if not polygon:
        return ""
    return ";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in polygon])


def apply_gate_to_data(data, x_channel, y_channel, polygon, parent_mask=None):
    """
    Applique un gate polygonal aux données et retourne le masque résultant.
    C'est LA fonction clé qui calcule quels événements sont dans le gate.
    """
    if polygon is None or len(polygon) < 3:
        # Si pas de polygone, retourner un masque vide
        return pd.Series(False, index=data.index)
    
    # Partir du masque parent ou de tous les événements
    if parent_mask is not None:
        base_mask = parent_mask.copy()
    else:
        base_mask = pd.Series(True, index=data.index)
    
    # Extraire les données
    x_data = data[x_channel].values
    y_data = data[y_channel].values
    
    # Filtrer les données valides
    valid = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0) & base_mask.values
    
    if not np.any(valid):
        return pd.Series(False, index=data.index)
    
    # Transformer les données
    x_transformed = biex_transform(x_data)
    y_transformed = biex_transform(y_data)
    
    # Préparer le polygone
    poly_x = np.array([p[0] for p in polygon])
    poly_y = np.array([p[1] for p in polygon])
    
    # Calculer quels points sont dans le polygone
    in_polygon = point_in_polygon(x_transformed, y_transformed, poly_x, poly_y)
    
    # Créer le masque final
    result_mask = pd.Series(False, index=data.index)
    result_mask.iloc[valid & in_polygon] = True
    
    return result_mask


def auto_gate_gmm(data, x_channel, y_channel, parent_mask=None, n_components=2, gate_type='main'):
    """Auto-gating avec GMM"""
    
    if parent_mask is not None:
        working_data = data[parent_mask]
    else:
        working_data = data
    
    x_data = working_data[x_channel].values
    y_data = working_data[y_channel].values
    
    valid = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
    x_valid = x_data[valid]
    y_valid = y_data[valid]
    
    if len(x_valid) < 100:
        return None
    
    x_t = biex_transform(x_valid)
    y_t = biex_transform(y_valid)
    
    X = np.column_stack([x_t, y_t])
    
    try:
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', 
                              random_state=42, n_init=3)
        gmm.fit(X)
        labels = gmm.predict(X)
        
        # Sélectionner le cluster selon le type
        if gate_type == 'main':
            cluster_sizes = [np.sum(labels == i) for i in range(n_components)]
            target_cluster = np.argmax(cluster_sizes)
        elif gate_type == 'low_y':
            cluster_means = [np.mean(y_t[labels == i]) for i in range(n_components)]
            target_cluster = np.argmin(cluster_means)
        elif gate_type == 'high_y':
            cluster_means = [np.mean(y_t[labels == i]) for i in range(n_components)]
            target_cluster = np.argmax(cluster_means)
        elif gate_type == 'high_x':
            cluster_means = [np.mean(x_t[labels == i]) for i in range(n_components)]
            target_cluster = np.argmax(cluster_means)
        elif gate_type == 'high_x_low_y':
            # Pour T cells: CD3 high, CD19 low
            scores = []
            for i in range(n_components):
                x_mean = np.mean(x_t[labels == i])
                y_mean = np.mean(y_t[labels == i])
                scores.append(x_mean - y_mean)
            target_cluster = np.argmax(scores)
        elif gate_type == 'high_x_low_y_v2':
            # Pour CD4+ T cells: CD4 high, CD8 low
            scores = []
            for i in range(n_components):
                x_mean = np.mean(x_t[labels == i])
                y_mean = np.mean(y_t[labels == i])
                scores.append(x_mean - y_mean)
            target_cluster = np.argmax(scores)
        else:
            target_cluster = 0
        
        cluster_mask = labels == target_cluster
        cluster_x = x_t[cluster_mask]
        cluster_y = y_t[cluster_mask]
        
        if len(cluster_x) < 30:
            return None
        
        # Convex Hull avec marge
        points = np.column_stack([cluster_x, cluster_y])
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        center = np.mean(hull_points, axis=0)
        margin = 1.08
        polygon = []
        for p in hull_points:
            expanded = center + margin * (p - center)
            polygon.append([float(expanded[0]), float(expanded[1])])
        
        # Simplifier si trop de points
        if len(polygon) > 15:
            step = max(1, len(polygon) // 12)
            polygon = polygon[::step]
        
        return polygon
        
    except Exception as e:
        return None


def apply_learned_adjustment(polygon, gate_name):
    """Applique les ajustements appris"""
    if polygon is None:
        return None
    
    params = load_learned_params()
    
    if gate_name in params['gates']:
        adj = params['gates'][gate_name]['avg_adjustment']
        dx = adj.get('x', 0)
        dy = adj.get('y', 0)
        
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            return [[p[0] + dx, p[1] + dy] for p in polygon]
    
    return polygon


def create_plot(data, x_channel, y_channel, x_marker, y_marker, title, 
                polygon, parent_mask, gate_name, n_in_gate, pct):
    """Crée le graphique Plotly"""
    
    if parent_mask is not None:
        plot_data = data[parent_mask]
    else:
        plot_data = data
    
    if len(plot_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    
    x_data = plot_data[x_channel].values
    y_data = plot_data[y_channel].values
    
    valid = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
    x_valid = x_data[valid]
    y_valid = y_data[valid]
    
    if len(x_valid) == 0:
        fig = go.Figure()
        return fig
    
    x_plot = biex_transform(x_valid)
    y_plot = biex_transform(y_valid)
    n_total = len(x_valid)
    
    # Sous-échantillonner
    max_pts = 10000
    if len(x_plot) > max_pts:
        idx = np.random.choice(len(x_plot), max_pts, replace=False)
        x_display, y_display = x_plot[idx], y_plot[idx]
    else:
        x_display, y_display = x_plot, y_plot
    
    fig = go.Figure()
    
    # Points
    fig.add_trace(go.Scattergl(
        x=x_display, y=y_display,
        mode='markers',
        marker=dict(size=3, color=y_display, colorscale='Viridis', opacity=0.5, showscale=False),
        hoverinfo='skip'
    ))
    
    # Polygone
    if polygon and len(polygon) >= 3:
        poly_x = [p[0] for p in polygon] + [polygon[0][0]]
        poly_y = [p[1] for p in polygon] + [polygon[0][1]]
        
        fig.add_trace(go.Scatter(
            x=poly_x, y=poly_y,
            mode='lines',
            line=dict(color='red', width=2.5),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            name=gate_name
        ))
        
        # Annotation
        cx, cy = np.mean(poly_x[:-1]), np.mean(poly_y[:-1])
        fig.add_annotation(
            x=cx, y=cy,
            text=f"<b>{gate_name}</b><br>{pct:.1f}%<br>({n_in_gate:,})",
            showarrow=False,
            font=dict(size=10, color="darkred"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="red", borderwidth=1, borderpad=3
        )
    
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sup>Parent: {n_total:,}</sup>", x=0.5, font=dict(size=12)),
        xaxis_title=f"<b>{x_marker}</b>",
        yaxis_title=f"<b>{y_marker}</b>",
        showlegend=False,
        width=450, height=380,
        margin=dict(l=50, r=20, t=55, b=45),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#eee', zeroline=False, showline=True, linecolor='black'),
        yaxis=dict(showgrid=True, gridcolor='#eee', zeroline=False, showline=True, linecolor='black'),
    )
    
    return fig


# ==================== INTERFACE STREAMLIT ====================

st.markdown('<h1 class="main-header">🔬 FACS Auto-Gating Intelligent</h1>', unsafe_allow_html=True)

# Charger paramètres appris
learned_params = load_learned_params()
n_learned = learned_params.get('n_corrections', 0)

if n_learned > 0:
    st.markdown(f"""
    <div class="learning-box">
    🧠 <b>Apprentissage actif</b> : {n_learned} correction(s) mémorisée(s) — L'auto-gating s'améliore !
    </div>
    """, unsafe_allow_html=True)

# Session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'reader' not in st.session_state:
    st.session_state.reader = None
if 'channels' not in st.session_state:
    st.session_state.channels = {}
if 'polygons' not in st.session_state:
    st.session_state.polygons = {}
if 'original_polygons' not in st.session_state:
    st.session_state.original_polygons = {}
if 'auto_done' not in st.session_state:
    st.session_state.auto_done = False

# Upload
uploaded_file = st.file_uploader("📁 Fichier FCS", type=['fcs'])

if uploaded_file:
    # Charger si nouveau fichier
    if st.session_state.reader is None or st.session_state.get('fname') != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        with st.spinner("Chargement..."):
            reader = FCSReader(tmp_path)
            st.session_state.reader = reader
            st.session_state.data = reader.data
            st.session_state.fname = uploaded_file.name
            st.session_state.polygons = {}
            st.session_state.original_polygons = {}
            st.session_state.auto_done = False
            
            # Identifier les canaux
            data = reader.data
            st.session_state.channels = {
                'FSC-A': find_channel(data, ['FSC-A']),
                'FSC-H': find_channel(data, ['FSC-H']),
                'SSC-A': find_channel(data, ['SSC-A']),
                'LiveDead': find_channel(data, ['LiveDead', 'Viab', 'Aqua']),
                'hCD45': find_channel(data, ['PerCP']),
                'CD3': find_channel(data, ['AF488', 'CD3']),
                'CD19': find_channel(data, ['PE-Fire700', 'CD19']),
                'CD4': find_channel(data, ['BV650', 'CD4']),
                'CD8': find_channel(data, ['BUV805', 'CD8']),
            }
    
    reader = st.session_state.reader
    data = st.session_state.data
    channels = st.session_state.channels
    n_total = len(data)
    
    # Métriques
    c1, c2, c3 = st.columns(3)
    c1.metric("Événements", f"{n_total:,}")
    c2.metric("Canaux", len(reader.channels))
    c3.metric("Fichier", reader.filename[:30])
    
    st.markdown("---")
    
    # ==================== AUTO-GATING ====================
    
    if not st.session_state.auto_done:
        if st.button("🚀 **LANCER L'AUTO-GATING**", type="primary", use_container_width=True):
            with st.spinner("Auto-gating en cours..."):
                prog = st.progress(0)
                
                # 1. Cells
                if channels['FSC-A'] and channels['SSC-A']:
                    poly = auto_gate_gmm(data, channels['FSC-A'], channels['SSC-A'], 
                                        parent_mask=None, n_components=2, gate_type='main')
                    poly = apply_learned_adjustment(poly, 'cells')
                    st.session_state.polygons['cells'] = poly
                    st.session_state.original_polygons['cells'] = [p.copy() for p in poly] if poly else None
                prog.progress(20)
                
                # 2. Singlets
                if channels['FSC-A'] and channels['FSC-H']:
                    cells_mask = apply_gate_to_data(data, channels['FSC-A'], channels['SSC-A'], 
                                                    st.session_state.polygons.get('cells'))
                    poly = auto_gate_gmm(data, channels['FSC-A'], channels['FSC-H'],
                                        parent_mask=cells_mask, n_components=2, gate_type='main')
                    poly = apply_learned_adjustment(poly, 'singlets')
                    st.session_state.polygons['singlets'] = poly
                    st.session_state.original_polygons['singlets'] = [p.copy() for p in poly] if poly else None
                prog.progress(40)
                
                # 3. Live
                if channels['LiveDead'] and channels['SSC-A']:
                    singlets_mask = apply_gate_to_data(data, channels['FSC-A'], channels['FSC-H'],
                                                       st.session_state.polygons.get('singlets'),
                                                       parent_mask=cells_mask)
                    poly = auto_gate_gmm(data, channels['LiveDead'], channels['SSC-A'],
                                        parent_mask=singlets_mask, n_components=2, gate_type='low_y')
                    # Pour Live/Dead, on veut la région LOW (négatif)
                    if poly:
                        # Ajuster le polygone vers la gauche (valeurs basses de Live/Dead)
                        poly = apply_learned_adjustment(poly, 'live')
                    st.session_state.polygons['live'] = poly
                    st.session_state.original_polygons['live'] = [p.copy() for p in poly] if poly else None
                prog.progress(60)
                
                # 4. hCD45+
                if channels['hCD45'] and channels['SSC-A']:
                    live_mask = apply_gate_to_data(data, channels['LiveDead'], channels['SSC-A'],
                                                   st.session_state.polygons.get('live'),
                                                   parent_mask=singlets_mask)
                    poly = auto_gate_gmm(data, channels['hCD45'], channels['SSC-A'],
                                        parent_mask=live_mask, n_components=2, gate_type='high_x')
                    poly = apply_learned_adjustment(poly, 'hcd45')
                    st.session_state.polygons['hcd45'] = poly
                    st.session_state.original_polygons['hcd45'] = [p.copy() for p in poly] if poly else None
                prog.progress(80)
                
                # 5. T cells (CD3+ CD19-)
                if channels['CD3'] and channels['CD19']:
                    hcd45_mask = apply_gate_to_data(data, channels['hCD45'], channels['SSC-A'],
                                                    st.session_state.polygons.get('hcd45'),
                                                    parent_mask=live_mask)
                    poly = auto_gate_gmm(data, channels['CD3'], channels['CD19'],
                                        parent_mask=hcd45_mask, n_components=3, gate_type='high_x_low_y')
                    poly = apply_learned_adjustment(poly, 't_cells')
                    st.session_state.polygons['t_cells'] = poly
                    st.session_state.original_polygons['t_cells'] = [p.copy() for p in poly] if poly else None
                
                # 6. CD4+ T cells
                if channels['CD4'] and channels['CD8']:
                    t_mask = apply_gate_to_data(data, channels['CD3'], channels['CD19'],
                                                st.session_state.polygons.get('t_cells'),
                                                parent_mask=hcd45_mask)
                    poly = auto_gate_gmm(data, channels['CD4'], channels['CD8'],
                                        parent_mask=t_mask, n_components=3, gate_type='high_x_low_y_v2')
                    poly = apply_learned_adjustment(poly, 'cd4_cells')
                    st.session_state.polygons['cd4_cells'] = poly
                    st.session_state.original_polygons['cd4_cells'] = [p.copy() for p in poly] if poly else None
                prog.progress(100)
                
                st.session_state.auto_done = True
                st.rerun()
    
    # ==================== AFFICHAGE & ÉDITION ====================
    
    if st.session_state.auto_done:
        
        st.markdown("""
        <div class="info-box">
        ✏️ <b>Modifiez les coordonnées</b> ci-dessous pour ajuster les gates. 
        Les populations enfants seront <b>automatiquement recalculées</b>.
        </div>
        """, unsafe_allow_html=True)
        
        # Récupérer les polygones (potentiellement modifiés)
        polygons = st.session_state.polygons
        
        # ==================== CALCUL EN CASCADE ====================
        # On recalcule TOUS les masques à chaque affichage pour prendre en compte les modifications
        
        all_stats = []
        
        # === NIVEAU 1: CELLS ===
        cells_mask = apply_gate_to_data(
            data, channels['FSC-A'], channels['SSC-A'], 
            polygons.get('cells'), parent_mask=None
        )
        n_cells = cells_mask.sum()
        pct_cells = n_cells / n_total * 100 if n_total > 0 else 0
        
        # === NIVEAU 2: SINGLETS ===
        singlets_mask = apply_gate_to_data(
            data, channels['FSC-A'], channels['FSC-H'],
            polygons.get('singlets'), parent_mask=cells_mask
        )
        n_singlets = singlets_mask.sum()
        pct_singlets = n_singlets / n_cells * 100 if n_cells > 0 else 0
        
        # === NIVEAU 3: LIVE ===
        live_mask = apply_gate_to_data(
            data, channels['LiveDead'], channels['SSC-A'],
            polygons.get('live'), parent_mask=singlets_mask
        )
        n_live = live_mask.sum()
        pct_live = n_live / n_singlets * 100 if n_singlets > 0 else 0
        
        # === NIVEAU 4: hCD45+ ===
        hcd45_mask = apply_gate_to_data(
            data, channels['hCD45'], channels['SSC-A'],
            polygons.get('hcd45'), parent_mask=live_mask
        )
        n_hcd45 = hcd45_mask.sum()
        pct_hcd45 = n_hcd45 / n_live * 100 if n_live > 0 else 0
        
        # === NIVEAU 5: T CELLS ===
        t_mask = apply_gate_to_data(
            data, channels['CD3'], channels['CD19'],
            polygons.get('t_cells'), parent_mask=hcd45_mask
        )
        n_tcells = t_mask.sum()
        pct_tcells = n_tcells / n_hcd45 * 100 if n_hcd45 > 0 else 0
        
        # === NIVEAU 6: CD4+ T CELLS ===
        cd4_mask = apply_gate_to_data(
            data, channels['CD4'], channels['CD8'],
            polygons.get('cd4_cells'), parent_mask=t_mask
        )
        n_cd4 = cd4_mask.sum()
        pct_cd4 = n_cd4 / n_tcells * 100 if n_tcells > 0 else 0
        
        # Collecter les stats
        all_stats = [
            {'Population': 'Cells', 'Parent': 'Ungated', 'Count': n_cells, '% Parent': round(pct_cells, 1)},
            {'Population': 'Single Cells', 'Parent': 'Cells', 'Count': n_singlets, '% Parent': round(pct_singlets, 1)},
            {'Population': 'Live', 'Parent': 'Single Cells', 'Count': n_live, '% Parent': round(pct_live, 1)},
            {'Population': 'hCD45+', 'Parent': 'Live', 'Count': n_hcd45, '% Parent': round(pct_hcd45, 1)},
            {'Population': 'T cells', 'Parent': 'hCD45+', 'Count': n_tcells, '% Parent': round(pct_tcells, 1)},
            {'Population': 'CD4+ T cells', 'Parent': 'T cells', 'Count': n_cd4, '% Parent': round(pct_cd4, 1)},
        ]
        
        # ==================== AFFICHAGE DES GRAPHIQUES ====================
        
        st.markdown("### 1️⃣ Gating Principal")
        
        col1, col2 = st.columns(2)
        
        # --- CELLS ---
        with col1:
            st.markdown("##### Cells")
            fig = create_plot(data, channels['FSC-A'], channels['SSC-A'], 'FSC-A', 'SSC-A',
                             f"Ungated → Cells", polygons.get('cells'), None, "Cells", n_cells, pct_cells)
            st.plotly_chart(fig, use_container_width=True, key="p_cells")
            
            new_coords = st.text_input(
                "Coordonnées Cells (x,y;x,y;...)", 
                value=format_coordinates(polygons.get('cells')),
                key="c_cells"
            )
            parsed = parse_coordinates(new_coords)
            if parsed and parsed != polygons.get('cells'):
                st.session_state.polygons['cells'] = parsed
                st.rerun()
        
        # --- SINGLETS ---
        with col2:
            st.markdown("##### Single Cells")
            fig = create_plot(data, channels['FSC-A'], channels['FSC-H'], 'FSC-A', 'FSC-H',
                             f"Cells → Singlets", polygons.get('singlets'), cells_mask, "Singlets", n_singlets, pct_singlets)
            st.plotly_chart(fig, use_container_width=True, key="p_sing")
            
            new_coords = st.text_input(
                "Coordonnées Singlets",
                value=format_coordinates(polygons.get('singlets')),
                key="c_sing"
            )
            parsed = parse_coordinates(new_coords)
            if parsed and parsed != polygons.get('singlets'):
                st.session_state.polygons['singlets'] = parsed
                st.rerun()
        
        col3, col4 = st.columns(2)
        
        # --- LIVE ---
        with col3:
            st.markdown("##### Live")
            fig = create_plot(data, channels['LiveDead'], channels['SSC-A'], 'Live/Dead', 'SSC-A',
                             f"Singlets → Live", polygons.get('live'), singlets_mask, "Live", n_live, pct_live)
            st.plotly_chart(fig, use_container_width=True, key="p_live")
            
            new_coords = st.text_input(
                "Coordonnées Live",
                value=format_coordinates(polygons.get('live')),
                key="c_live"
            )
            parsed = parse_coordinates(new_coords)
            if parsed and parsed != polygons.get('live'):
                st.session_state.polygons['live'] = parsed
                st.rerun()
        
        # --- hCD45+ ---
        with col4:
            st.markdown("##### hCD45+")
            fig = create_plot(data, channels['hCD45'], channels['SSC-A'], 'hCD45', 'SSC-A',
                             f"Live → hCD45+", polygons.get('hcd45'), live_mask, "hCD45+", n_hcd45, pct_hcd45)
            st.plotly_chart(fig, use_container_width=True, key="p_hcd45")
            
            new_coords = st.text_input(
                "Coordonnées hCD45+",
                value=format_coordinates(polygons.get('hcd45')),
                key="c_hcd45"
            )
            parsed = parse_coordinates(new_coords)
            if parsed and parsed != polygons.get('hcd45'):
                st.session_state.polygons['hcd45'] = parsed
                st.rerun()
        
        # ==================== SOUS-POPULATIONS ====================
        
        st.markdown("### 2️⃣ Sous-Populations")
        
        col5, col6 = st.columns(2)
        
        # --- T CELLS ---
        with col5:
            st.markdown("##### T cells (CD3+ CD19-)")
            fig = create_plot(data, channels['CD3'], channels['CD19'], 'CD3', 'CD19',
                             f"hCD45+ → T cells", polygons.get('t_cells'), hcd45_mask, "T cells", n_tcells, pct_tcells)
            st.plotly_chart(fig, use_container_width=True, key="p_tcells")
            
            new_coords = st.text_input(
                "Coordonnées T cells",
                value=format_coordinates(polygons.get('t_cells')),
                key="c_tcells"
            )
            parsed = parse_coordinates(new_coords)
            if parsed and parsed != polygons.get('t_cells'):
                st.session_state.polygons['t_cells'] = parsed
                st.rerun()
        
        # --- CD4+ T CELLS ---
        with col6:
            st.markdown("##### CD4+ T cells")
            if n_tcells > 0:
                fig = create_plot(data, channels['CD4'], channels['CD8'], 'CD4', 'CD8',
                                 f"T cells → CD4+", polygons.get('cd4_cells'), t_mask, "CD4+", n_cd4, pct_cd4)
                st.plotly_chart(fig, use_container_width=True, key="p_cd4")
                
                new_coords = st.text_input(
                    "Coordonnées CD4+",
                    value=format_coordinates(polygons.get('cd4_cells')),
                    key="c_cd4"
                )
                parsed = parse_coordinates(new_coords)
                if parsed and parsed != polygons.get('cd4_cells'):
                    st.session_state.polygons['cd4_cells'] = parsed
                    st.rerun()
            else:
                st.info("Ajustez le gate T cells d'abord")
        
        # ==================== ACTIONS ====================
        
        st.markdown("---")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("🔄 Rafraîchir", use_container_width=True):
                st.rerun()
        
        with col_b:
            if st.button("💾 **Sauvegarder corrections (apprentissage)**", type="primary", use_container_width=True):
                n_saved = 0
                for gname in st.session_state.polygons:
                    current = st.session_state.polygons.get(gname)
                    original = st.session_state.original_polygons.get(gname)
                    
                    if current and original:
                        # Vérifier si modifié
                        curr_str = format_coordinates(current)
                        orig_str = format_coordinates(original)
                        
                        if curr_str != orig_str:
                            update_learned_params(gname, original, current, {'filename': reader.filename})
                            n_saved += 1
                
                if n_saved > 0:
                    st.success(f"✅ {n_saved} correction(s) sauvegardée(s) ! L'auto-gating s'améliorera.")
                else:
                    st.info("Aucune modification détectée.")
        
        with col_c:
            if st.button("🔃 Réinitialiser", use_container_width=True):
                st.session_state.polygons = {k: [p.copy() for p in v] if v else None 
                                             for k, v in st.session_state.original_polygons.items()}
                st.rerun()
        
        # ==================== TABLEAU RÉCAPITULATIF ====================
        
        st.markdown("### 📊 Résumé des Populations")
        
        stats_df = pd.DataFrame(all_stats)
        stats_df['% Total'] = (stats_df['Count'] / n_total * 100).round(2)
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Export
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            csv = stats_df.to_csv(index=False)
            st.download_button("📥 Télécharger CSV", csv, f"{reader.filename}_stats.csv", "text/csv", use_container_width=True)
        
        with col_e2:
            try:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    stats_df.to_excel(w, sheet_name='Populations', index=False)
                buf.seek(0)
                st.download_button("📥 Télécharger Excel", buf, f"{reader.filename}_stats.xlsx", use_container_width=True)
            except:
                pass

else:
    st.markdown("""
    <div class="info-box">
    <h3>🔬 Auto-Gating Intelligent avec Apprentissage</h3>
    <ol>
    <li><b>Uploadez</b> un fichier FCS</li>
    <li><b>Lancez l'auto-gating</b> — L'algorithme identifie automatiquement les populations</li>
    <li><b>Vérifiez et corrigez</b> les gates si nécessaire (modifiez les coordonnées)</li>
    <li><b>Sauvegardez vos corrections</b> — L'algorithme apprend et s'améliore !</li>
    </ol>
    <p>Plus vous corrigez, plus l'auto-gating devient précis 🧠</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
🔬 FACS Auto-Gating Intelligent | 🧠 {n_learned} correction(s) apprises
</div>
""", unsafe_allow_html=True)
