#!/usr/bin/env python3
"""
FACS Autogating - Auto-Gating Intelligent avec Apprentissage
- Auto-gating automatique optimisé (GMM + heuristiques)
- Correction manuelle des gates (polygones éditables)
- Apprentissage des corrections pour améliorer les futurs auto-gatings
- Sauvegarde des paramètres optimisés
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from scipy import ndimage
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
    .main-header { font-size: 2rem; color: #2c3e50; text-align: center; }
    .success-box { background: #d4edda; padding: 0.8rem; border-radius: 0.5rem;
                   border-left: 4px solid #28a745; margin: 0.5rem 0; }
    .warning-box { background: #fff3cd; padding: 0.8rem; border-radius: 0.5rem;
                   border-left: 4px solid #ffc107; margin: 0.5rem 0; }
    .info-box { background: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem;
                border-left: 4px solid #0066cc; margin: 0.5rem 0; }
    .learning-box { background: #f0e6ff; padding: 0.8rem; border-radius: 0.5rem;
                    border-left: 4px solid #6f42c1; margin: 0.5rem 0; }
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
    return {
        'n_corrections': 0,
        'gates': {},
        'last_updated': None
    }


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
            'avg_adjustment': {'x': 0, 'y': 0, 'scale': 1.0},
            'n_samples': 0
        }
    
    gate_params = params['gates'][gate_name]
    
    # Calculer l'ajustement
    orig_center = np.mean(original_polygon, axis=0)
    corr_center = np.mean(corrected_polygon, axis=0)
    
    adjustment = {
        'dx': float(corr_center[0] - orig_center[0]),
        'dy': float(corr_center[1] - orig_center[1]),
        'data_stats': data_stats
    }
    
    gate_params['corrections'].append(adjustment)
    gate_params['n_samples'] += 1
    
    # Mettre à jour la moyenne des ajustements (moyenne mobile)
    n = gate_params['n_samples']
    alpha = 2 / (n + 1)  # Facteur de lissage exponentiel
    
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
    
    def get_marker(self, channel):
        return self.channel_markers.get(channel, channel)


def find_channel(data, keywords):
    for col in data.columns:
        col_upper = col.upper()
        for kw in keywords:
            if kw.upper() in col_upper:
                return col
    return None


def biex_transform(x):
    """Transformation biexponentielle simplifiée"""
    x = np.asarray(x, dtype=float)
    return np.arcsinh(x / 150) * 50


def biex_inverse(y):
    """Inverse de la transformation biexponentielle"""
    return np.sinh(y / 50) * 150


def auto_gate_gmm(x_data, y_data, n_components=2, gate_type='main_population'):
    """
    Auto-gating intelligent avec GMM
    Retourne un polygone optimal pour la population cible
    """
    # Filtrer les données valides
    valid = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
    x_valid = x_data[valid]
    y_valid = y_data[valid]
    
    if len(x_valid) < 100:
        return None, valid
    
    # Transformer
    x_t = biex_transform(x_valid)
    y_t = biex_transform(y_valid)
    
    # Préparer les données pour GMM
    X = np.column_stack([x_t, y_t])
    
    # Ajuster GMM
    try:
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', 
                              random_state=42, n_init=3)
        gmm.fit(X)
        labels = gmm.predict(X)
        
        # Sélectionner le cluster principal selon le type de gate
        if gate_type == 'main_population':
            # Prendre le cluster le plus dense (plus de points)
            cluster_sizes = [np.sum(labels == i) for i in range(n_components)]
            target_cluster = np.argmax(cluster_sizes)
        elif gate_type == 'high_fsc':
            # Prendre le cluster avec FSC le plus élevé
            cluster_means = [np.mean(x_t[labels == i]) for i in range(n_components)]
            target_cluster = np.argmax(cluster_means)
        elif gate_type == 'low_marker':
            # Prendre le cluster avec Y le plus bas (ex: Live/Dead négatif)
            cluster_means = [np.mean(y_t[labels == i]) for i in range(n_components)]
            target_cluster = np.argmin(cluster_means)
        elif gate_type == 'high_marker':
            # Prendre le cluster avec Y le plus élevé
            cluster_means = [np.mean(y_t[labels == i]) for i in range(n_components)]
            target_cluster = np.argmax(cluster_means)
        else:
            target_cluster = 0
        
        # Points du cluster cible
        cluster_mask = labels == target_cluster
        cluster_x = x_t[cluster_mask]
        cluster_y = y_t[cluster_mask]
        
        if len(cluster_x) < 50:
            return None, valid
        
        # Créer le polygone avec Convex Hull + marge
        points = np.column_stack([cluster_x, cluster_y])
        
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # Ajouter une marge de 5%
            center = np.mean(hull_points, axis=0)
            margin = 1.05
            polygon = []
            for p in hull_points:
                expanded = center + margin * (p - center)
                polygon.append([float(expanded[0]), float(expanded[1])])
            
            # Simplifier le polygone si trop de points
            if len(polygon) > 12:
                step = len(polygon) // 10
                polygon = polygon[::step]
            
            return polygon, cluster_mask
            
        except:
            # Fallback: ellipse basée sur la covariance
            return create_ellipse_polygon(cluster_x, cluster_y), cluster_mask
    
    except Exception as e:
        return None, valid


def auto_gate_density(x_data, y_data, percentile_threshold=85):
    """
    Auto-gating basé sur la densité
    Trouve la région de haute densité
    """
    valid = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
    x_valid = x_data[valid]
    y_valid = y_data[valid]
    
    if len(x_valid) < 100:
        return None
    
    x_t = biex_transform(x_valid)
    y_t = biex_transform(y_valid)
    
    # Calculer la densité 2D
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([x_t, y_t])
        kde = gaussian_kde(xy)
        density = kde(xy)
        
        # Seuil de densité
        threshold = np.percentile(density, 100 - percentile_threshold)
        high_density = density >= threshold
        
        # Points de haute densité
        hd_x = x_t[high_density]
        hd_y = y_t[high_density]
        
        if len(hd_x) < 50:
            return None
        
        # Convex Hull
        points = np.column_stack([hd_x, hd_y])
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        polygon = [[float(p[0]), float(p[1])] for p in hull_points]
        
        return polygon
        
    except:
        return None


def create_ellipse_polygon(x_data, y_data, n_points=20, n_std=2.5):
    """Crée un polygone elliptique basé sur la distribution des données"""
    center_x = np.mean(x_data)
    center_y = np.mean(y_data)
    std_x = np.std(x_data) * n_std
    std_y = np.std(y_data) * n_std
    
    # Si corrélation, ajuster l'ellipse
    if len(x_data) > 10:
        corr = np.corrcoef(x_data, y_data)[0, 1]
        if np.isfinite(corr):
            rotation = np.arctan2(corr * std_y, std_x) / 2
        else:
            rotation = 0
    else:
        rotation = 0
    
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    polygon = []
    
    for a in angles:
        # Point sur l'ellipse
        x = std_x * np.cos(a)
        y = std_y * np.sin(a)
        
        # Rotation
        x_rot = x * np.cos(rotation) - y * np.sin(rotation)
        y_rot = x * np.sin(rotation) + y * np.cos(rotation)
        
        polygon.append([float(center_x + x_rot), float(center_y + y_rot)])
    
    return polygon


def apply_learned_adjustment(polygon, gate_name):
    """Applique les ajustements appris au polygone"""
    params = load_learned_params()
    
    if gate_name in params['gates']:
        adj = params['gates'][gate_name]['avg_adjustment']
        dx = adj.get('x', 0)
        dy = adj.get('y', 0)
        
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            adjusted = [[p[0] + dx, p[1] + dy] for p in polygon]
            return adjusted
    
    return polygon


def point_in_polygon(x, y, poly_x, poly_y):
    """Vérifie si les points sont dans le polygone (ray casting)"""
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


def create_gating_plot(data, x_channel, y_channel, x_marker, y_marker, 
                       title, polygon, parent_mask=None, gate_name="Gate",
                       editable=True):
    """Crée un graphique avec gate polygonal éditable"""
    
    if parent_mask is not None:
        plot_data = data[parent_mask].copy()
        parent_indices = data.index[parent_mask]
    else:
        plot_data = data.copy()
        parent_indices = data.index
    
    if len(plot_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig, pd.Series(False, index=data.index), 0, 0
    
    x_data = plot_data[x_channel].values
    y_data = plot_data[y_channel].values
    
    valid = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
    x_valid = x_data[valid]
    y_valid = y_data[valid]
    valid_indices = parent_indices[valid]
    
    if len(x_valid) == 0:
        fig = go.Figure()
        return fig, pd.Series(False, index=data.index), 0, 0
    
    x_plot = biex_transform(x_valid)
    y_plot = biex_transform(y_valid)
    n_total = len(x_valid)
    
    # Sous-échantillonner
    max_points = 12000
    if len(x_plot) > max_points:
        idx = np.random.choice(len(x_plot), max_points, replace=False)
        x_display = x_plot[idx]
        y_display = y_plot[idx]
    else:
        x_display = x_plot
        y_display = y_plot
    
    fig = go.Figure()
    
    # Points
    fig.add_trace(go.Scattergl(
        x=x_display, y=y_display,
        mode='markers',
        marker=dict(size=3, color=y_display, colorscale='Viridis', opacity=0.6, showscale=False),
        name='Events',
        hoverinfo='skip'
    ))
    
    # Gate polygonal
    n_in_gate = 0
    pct = 0
    gate_mask = pd.Series(False, index=data.index)
    
    if polygon and len(polygon) >= 3:
        poly_x = [p[0] for p in polygon] + [polygon[0][0]]
        poly_y = [p[1] for p in polygon] + [polygon[0][1]]
        
        # Dessiner le polygone (shape éditable)
        if editable:
            # Ajouter comme shape éditable
            fig.add_shape(
                type="path",
                path="M " + " L ".join([f"{x},{y}" for x, y in zip(poly_x, poly_y)]) + " Z",
                line=dict(color="red", width=2),
                fillcolor="rgba(255, 0, 0, 0.1)",
                editable=True,
                name=gate_name
            )
        else:
            fig.add_trace(go.Scatter(
                x=poly_x, y=poly_y,
                mode='lines',
                line=dict(color='red', width=2),
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.1)',
                name=gate_name
            ))
        
        # Calculer les événements dans le gate
        in_poly = point_in_polygon(x_plot, y_plot, np.array(poly_x[:-1]), np.array(poly_y[:-1]))
        n_in_gate = in_poly.sum()
        pct = n_in_gate / n_total * 100 if n_total > 0 else 0
        
        gate_mask.loc[valid_indices[in_poly]] = True
        
        # Annotation
        cx = np.mean(poly_x[:-1])
        cy = np.mean(poly_y[:-1])
        fig.add_annotation(
            x=cx, y=cy,
            text=f"<b>{gate_name}</b><br>{pct:.1f}%<br>({n_in_gate:,})",
            showarrow=False,
            font=dict(size=11, color="darkred"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="red", borderwidth=2, borderpad=4
        )
    
    # Layout
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sup>n={n_total:,}</sup>", x=0.5),
        xaxis_title=f"<b>{x_marker}</b>",
        yaxis_title=f"<b>{y_marker}</b>",
        showlegend=False,
        width=480, height=420,
        dragmode='pan',
        margin=dict(l=50, r=20, t=60, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#eee', zeroline=False, showline=True, linecolor='black'),
        yaxis=dict(showgrid=True, gridcolor='#eee', zeroline=False, showline=True, linecolor='black'),
        modebar=dict(add=['drawclosedpath', 'eraseshape'], remove=['lasso2d', 'select2d']),
        newshape=dict(line=dict(color='red', width=2), fillcolor='rgba(255,0,0,0.1)')
    )
    
    return fig, gate_mask, n_in_gate, pct


# ==================== INTERFACE STREAMLIT ====================

st.markdown('<h1 class="main-header">🔬 FACS Auto-Gating Intelligent</h1>', unsafe_allow_html=True)

# Charger les paramètres appris
learned_params = load_learned_params()
n_corrections = learned_params.get('n_corrections', 0)

# Afficher le statut d'apprentissage
if n_corrections > 0:
    st.markdown(f"""
    <div class="learning-box">
    🧠 <b>Apprentissage actif</b> : {n_corrections} correction(s) mémorisée(s)<br>
    <small>L'auto-gating s'améliore avec vos corrections</small>
    </div>
    """, unsafe_allow_html=True)

# Session state
if 'polygons' not in st.session_state:
    st.session_state.polygons = {}
if 'original_polygons' not in st.session_state:
    st.session_state.original_polygons = {}
if 'masks' not in st.session_state:
    st.session_state.masks = {}
if 'reader' not in st.session_state:
    st.session_state.reader = None
if 'auto_gating_done' not in st.session_state:
    st.session_state.auto_gating_done = False

# Upload
uploaded_file = st.file_uploader("📁 Fichier FCS", type=['fcs'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    try:
        # Charger les données
        if st.session_state.reader is None or st.session_state.get('filename') != uploaded_file.name:
            with st.spinner("Chargement..."):
                reader = FCSReader(tmp_path)
                st.session_state.reader = reader
                st.session_state.filename = uploaded_file.name
                st.session_state.polygons = {}
                st.session_state.original_polygons = {}
                st.session_state.masks = {}
                st.session_state.auto_gating_done = False
        
        reader = st.session_state.reader
        data = reader.data
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        col1.metric("Événements", f"{len(data):,}")
        col2.metric("Canaux", len(reader.channels))
        col3.metric("Fichier", reader.filename[:25])
        
        # Trouver les canaux
        channels = {
            'FSC-A': find_channel(data, ['FSC-A']),
            'FSC-H': find_channel(data, ['FSC-H']),
            'SSC-A': find_channel(data, ['SSC-A']),
            'LiveDead': find_channel(data, ['LiveDead', 'Viab']),
            'hCD45': find_channel(data, ['PerCP-A']),
            'CD3': find_channel(data, ['AF488', 'CD3']),
            'CD19': find_channel(data, ['PE-Fire700', 'CD19']),
            'CD4': find_channel(data, ['BV650', 'CD4']),
            'CD8': find_channel(data, ['BUV805', 'CD8']),
            'CD56': find_channel(data, ['PE-Cy7', 'CD56']),
            'CD16': find_channel(data, ['NovaFluor', 'CD16']),
            'FoxP3': find_channel(data, ['eFluor450', 'FoxP3']),
            'CD25': find_channel(data, ['BV785', 'CD25']),
        }
        
        st.markdown("---")
        
        # ==================== AUTO-GATING ====================
        
        if not st.session_state.auto_gating_done:
            if st.button("🚀 **Lancer l'Auto-Gating**", type="primary", use_container_width=True):
                with st.spinner("Auto-gating en cours..."):
                    progress = st.progress(0)
                    
                    # Gate 1: Cells (FSC-A vs SSC-A)
                    if channels['FSC-A'] and channels['SSC-A']:
                        polygon, _ = auto_gate_gmm(
                            data[channels['FSC-A']].values, 
                            data[channels['SSC-A']].values,
                            n_components=2, gate_type='main_population'
                        )
                        if polygon:
                            polygon = apply_learned_adjustment(polygon, 'cells')
                            st.session_state.polygons['cells'] = polygon
                            st.session_state.original_polygons['cells'] = polygon.copy()
                    progress.progress(15)
                    
                    # Gate 2: Singlets (FSC-A vs FSC-H)
                    if channels['FSC-A'] and channels['FSC-H']:
                        polygon, _ = auto_gate_gmm(
                            data[channels['FSC-A']].values,
                            data[channels['FSC-H']].values,
                            n_components=2, gate_type='main_population'
                        )
                        if polygon:
                            polygon = apply_learned_adjustment(polygon, 'singlets')
                            st.session_state.polygons['singlets'] = polygon
                            st.session_state.original_polygons['singlets'] = polygon.copy()
                    progress.progress(30)
                    
                    # Gate 3: Live (Live/Dead négatif)
                    if channels['LiveDead'] and channels['SSC-A']:
                        polygon, _ = auto_gate_gmm(
                            data[channels['LiveDead']].values,
                            data[channels['SSC-A']].values,
                            n_components=2, gate_type='low_marker'
                        )
                        if polygon:
                            polygon = apply_learned_adjustment(polygon, 'live')
                            st.session_state.polygons['live'] = polygon
                            st.session_state.original_polygons['live'] = polygon.copy()
                    progress.progress(45)
                    
                    # Gate 4: hCD45+
                    if channels['hCD45'] and channels['SSC-A']:
                        polygon, _ = auto_gate_gmm(
                            data[channels['hCD45']].values,
                            data[channels['SSC-A']].values,
                            n_components=2, gate_type='high_marker'
                        )
                        if polygon:
                            polygon = apply_learned_adjustment(polygon, 'hcd45')
                            st.session_state.polygons['hcd45'] = polygon
                            st.session_state.original_polygons['hcd45'] = polygon.copy()
                    progress.progress(60)
                    
                    # Gate 5: T cells (CD3+ CD19-)
                    if channels['CD3'] and channels['CD19']:
                        polygon, _ = auto_gate_gmm(
                            data[channels['CD3']].values,
                            data[channels['CD19']].values,
                            n_components=3, gate_type='high_marker'
                        )
                        if polygon:
                            polygon = apply_learned_adjustment(polygon, 't_cells')
                            st.session_state.polygons['t_cells'] = polygon
                            st.session_state.original_polygons['t_cells'] = polygon.copy()
                    progress.progress(75)
                    
                    # Gate 6: CD4+ T cells
                    if channels['CD4'] and channels['CD8']:
                        polygon, _ = auto_gate_gmm(
                            data[channels['CD4']].values,
                            data[channels['CD8']].values,
                            n_components=3, gate_type='high_marker'
                        )
                        if polygon:
                            polygon = apply_learned_adjustment(polygon, 'cd4_cells')
                            st.session_state.polygons['cd4_cells'] = polygon
                            st.session_state.original_polygons['cd4_cells'] = polygon.copy()
                    progress.progress(100)
                    
                    st.session_state.auto_gating_done = True
                    st.rerun()
        
        # ==================== AFFICHAGE DES GATES ====================
        
        if st.session_state.auto_gating_done:
            
            st.markdown("""
            <div class="info-box">
            📌 <b>Correction des gates</b> : Modifiez les coordonnées dans les champs ci-dessous pour ajuster chaque gate.
            Vos corrections seront mémorisées pour améliorer les futurs auto-gatings.
            </div>
            """, unsafe_allow_html=True)
            
            all_stats = []
            n_total = len(data)
            
            # ===== ROW 1 =====
            st.markdown("### 1️⃣ Gating Principal")
            col1, col2 = st.columns(2)
            
            # Plot 1: Cells
            with col1:
                st.markdown("##### Cells (FSC-A vs SSC-A)")
                if channels['FSC-A'] and channels['SSC-A']:
                    polygon = st.session_state.polygons.get('cells', [])
                    
                    fig, mask, n_gate, pct = create_gating_plot(
                        data, channels['FSC-A'], channels['SSC-A'], 'FSC-A', 'SSC-A',
                        f"Ungated (n={n_total:,})", polygon, gate_name="Cells"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="plot_cells")
                    
                    # Champ d'édition des coordonnées
                    coords_str = st.text_input(
                        "Coordonnées (x1,y1;x2,y2;...)",
                        value=";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in polygon]) if polygon else "",
                        key="coords_cells"
                    )
                    if coords_str:
                        try:
                            new_coords = [[float(c) for c in p.split(',')] for p in coords_str.split(';') if ',' in p]
                            if len(new_coords) >= 3 and new_coords != st.session_state.polygons.get('cells'):
                                st.session_state.polygons['cells'] = new_coords
                        except:
                            pass
                    
                    st.session_state.masks['cells'] = mask
                    if mask.sum() > 0:
                        all_stats.append({'Population': 'Cells', 'Parent': 'Ungated',
                                         'Count': mask.sum(), '% Parent': round(mask.sum()/n_total*100, 1)})
            
            # Plot 2: Singlets
            with col2:
                st.markdown("##### Single Cells (FSC-A vs FSC-H)")
                if channels['FSC-A'] and channels['FSC-H']:
                    parent = st.session_state.masks.get('cells')
                    n_parent = parent.sum() if parent is not None and parent.sum() > 0 else n_total
                    polygon = st.session_state.polygons.get('singlets', [])
                    
                    fig, mask, n_gate, pct = create_gating_plot(
                        data, channels['FSC-A'], channels['FSC-H'], 'FSC-A', 'FSC-H',
                        f"Cells (n={n_parent:,})", polygon, parent_mask=parent, gate_name="Singlets"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="plot_singlets")
                    
                    coords_str = st.text_input(
                        "Coordonnées",
                        value=";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in polygon]) if polygon else "",
                        key="coords_singlets"
                    )
                    if coords_str:
                        try:
                            new_coords = [[float(c) for c in p.split(',')] for p in coords_str.split(';') if ',' in p]
                            if len(new_coords) >= 3:
                                st.session_state.polygons['singlets'] = new_coords
                        except:
                            pass
                    
                    st.session_state.masks['singlets'] = mask
                    if mask.sum() > 0:
                        all_stats.append({'Population': 'Single Cells', 'Parent': 'Cells',
                                         'Count': mask.sum(), '% Parent': round(mask.sum()/n_parent*100, 1)})
            
            # ===== ROW 2 =====
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("##### Live (Live/Dead négatif)")
                if channels['LiveDead'] and channels['SSC-A']:
                    parent = st.session_state.masks.get('singlets')
                    n_parent = parent.sum() if parent is not None and parent.sum() > 0 else n_total
                    polygon = st.session_state.polygons.get('live', [])
                    
                    fig, mask, n_gate, pct = create_gating_plot(
                        data, channels['LiveDead'], channels['SSC-A'], 'Live/Dead', 'SSC-A',
                        f"Singlets (n={n_parent:,})", polygon, parent_mask=parent, gate_name="Live"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="plot_live")
                    
                    coords_str = st.text_input(
                        "Coordonnées",
                        value=";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in polygon]) if polygon else "",
                        key="coords_live"
                    )
                    if coords_str:
                        try:
                            new_coords = [[float(c) for c in p.split(',')] for p in coords_str.split(';') if ',' in p]
                            if len(new_coords) >= 3:
                                st.session_state.polygons['live'] = new_coords
                        except:
                            pass
                    
                    st.session_state.masks['live'] = mask
                    if mask.sum() > 0:
                        all_stats.append({'Population': 'Live', 'Parent': 'Single Cells',
                                         'Count': mask.sum(), '% Parent': round(mask.sum()/n_parent*100, 1)})
            
            with col4:
                st.markdown("##### hCD45+ (Leucocytes)")
                if channels['hCD45'] and channels['SSC-A']:
                    parent = st.session_state.masks.get('live')
                    n_parent = parent.sum() if parent is not None and parent.sum() > 0 else n_total
                    polygon = st.session_state.polygons.get('hcd45', [])
                    
                    fig, mask, n_gate, pct = create_gating_plot(
                        data, channels['hCD45'], channels['SSC-A'], 'hCD45', 'SSC-A',
                        f"Live (n={n_parent:,})", polygon, parent_mask=parent, gate_name="hCD45+"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="plot_hcd45")
                    
                    coords_str = st.text_input(
                        "Coordonnées",
                        value=";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in polygon]) if polygon else "",
                        key="coords_hcd45"
                    )
                    if coords_str:
                        try:
                            new_coords = [[float(c) for c in p.split(',')] for p in coords_str.split(';') if ',' in p]
                            if len(new_coords) >= 3:
                                st.session_state.polygons['hcd45'] = new_coords
                        except:
                            pass
                    
                    st.session_state.masks['hcd45'] = mask
                    if mask.sum() > 0:
                        all_stats.append({'Population': 'hCD45+', 'Parent': 'Live',
                                         'Count': mask.sum(), '% Parent': round(mask.sum()/n_parent*100, 1)})
            
            # ===== ROW 3: SOUS-POPULATIONS =====
            st.markdown("### 2️⃣ Sous-Populations")
            col5, col6 = st.columns(2)
            
            with col5:
                st.markdown("##### T cells (CD3+)")
                if channels['CD3'] and channels['CD19']:
                    parent = st.session_state.masks.get('hcd45')
                    n_parent = parent.sum() if parent is not None and parent.sum() > 0 else n_total
                    polygon = st.session_state.polygons.get('t_cells', [])
                    
                    fig, mask, n_gate, pct = create_gating_plot(
                        data, channels['CD3'], channels['CD19'], 'CD3', 'CD19',
                        f"hCD45+ (n={n_parent:,})", polygon, parent_mask=parent, gate_name="T cells"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="plot_tcells")
                    
                    coords_str = st.text_input(
                        "Coordonnées",
                        value=";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in polygon]) if polygon else "",
                        key="coords_tcells"
                    )
                    if coords_str:
                        try:
                            new_coords = [[float(c) for c in p.split(',')] for p in coords_str.split(';') if ',' in p]
                            if len(new_coords) >= 3:
                                st.session_state.polygons['t_cells'] = new_coords
                        except:
                            pass
                    
                    st.session_state.masks['t_cells'] = mask
                    if mask.sum() > 0:
                        all_stats.append({'Population': 'T cells', 'Parent': 'hCD45+',
                                         'Count': mask.sum(), '% Parent': round(mask.sum()/n_parent*100, 1)})
            
            with col6:
                st.markdown("##### CD4+ T cells")
                if channels['CD4'] and channels['CD8']:
                    parent = st.session_state.masks.get('t_cells')
                    n_parent = parent.sum() if parent is not None and parent.sum() > 0 else 0
                    polygon = st.session_state.polygons.get('cd4_cells', [])
                    
                    if n_parent > 0:
                        fig, mask, n_gate, pct = create_gating_plot(
                            data, channels['CD4'], channels['CD8'], 'CD4', 'CD8',
                            f"T cells (n={n_parent:,})", polygon, parent_mask=parent, gate_name="CD4+"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="plot_cd4")
                        
                        coords_str = st.text_input(
                            "Coordonnées",
                            value=";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in polygon]) if polygon else "",
                            key="coords_cd4"
                        )
                        if coords_str:
                            try:
                                new_coords = [[float(c) for c in p.split(',')] for p in coords_str.split(';') if ',' in p]
                                if len(new_coords) >= 3:
                                    st.session_state.polygons['cd4_cells'] = new_coords
                            except:
                                pass
                        
                        st.session_state.masks['cd4_cells'] = mask
                        if mask.sum() > 0:
                            all_stats.append({'Population': 'CD4+ T cells', 'Parent': 'T cells',
                                             'Count': mask.sum(), '% Parent': round(mask.sum()/n_parent*100, 1)})
            
            # ==================== ACTIONS ====================
            
            st.markdown("---")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("🔄 **Recalculer**", use_container_width=True):
                    st.rerun()
            
            with col_b:
                if st.button("💾 **Sauvegarder corrections**", type="primary", use_container_width=True):
                    # Comparer les polygones actuels avec les originaux et sauvegarder
                    corrections_saved = 0
                    for gate_name in st.session_state.polygons:
                        current = st.session_state.polygons.get(gate_name, [])
                        original = st.session_state.original_polygons.get(gate_name, [])
                        
                        if current and original and current != original:
                            data_stats = {
                                'n_events': len(data),
                                'filename': reader.filename
                            }
                            update_learned_params(gate_name, original, current, data_stats)
                            corrections_saved += 1
                    
                    if corrections_saved > 0:
                        st.success(f"✅ {corrections_saved} correction(s) sauvegardée(s) ! L'auto-gating s'améliorera.")
                    else:
                        st.info("Aucune correction détectée.")
            
            with col_c:
                if st.button("🔃 **Réinitialiser**", use_container_width=True):
                    st.session_state.polygons = st.session_state.original_polygons.copy()
                    st.rerun()
            
            # ==================== STATISTIQUES ====================
            
            st.markdown("### 📊 Résumé des Populations")
            
            if all_stats:
                stats_df = pd.DataFrame(all_stats)
                stats_df['% Total'] = (stats_df['Count'] / n_total * 100).round(2)
                
                st.dataframe(stats_df, use_container_width=True)
                
                # Export
                col1, col2 = st.columns(2)
                with col1:
                    csv = stats_df.to_csv(index=False)
                    st.download_button("📥 CSV", csv, f"{reader.filename}_stats.csv", "text/csv", use_container_width=True)
                
                with col2:
                    try:
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            stats_df.to_excel(writer, sheet_name='Populations', index=False)
                        output.seek(0)
                        st.download_button("📥 Excel", output, f"{reader.filename}_stats.xlsx", use_container_width=True)
                    except:
                        pass
    
    except Exception as e:
        st.error(f"Erreur: {e}")
        st.exception(e)

else:
    st.markdown("""
    <div class="info-box">
    <b>🔬 Auto-Gating Intelligent</b><br><br>
    Cette application effectue un <b>auto-gating automatique</b> et <b>apprend de vos corrections</b> :
    <ol>
    <li>Uploadez un fichier FCS</li>
    <li>L'auto-gating identifie automatiquement les populations</li>
    <li>Corrigez les gates si nécessaire (modifiez les coordonnées)</li>
    <li>Sauvegardez vos corrections → l'algorithme s'améliore</li>
    </ol>
    Plus vous corrigez, plus l'auto-gating devient précis ! 🧠
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    🔬 <b>FACS Auto-Gating Intelligent</b> | 🧠 {n_corrections} correction(s) apprises
</div>
""", unsafe_allow_html=True)
