#!/usr/bin/env python3
"""
FACS Autogating - FlowJo Style v7
- Visualisation style FlowJo (contour, densité, pseudo-color)
- Populations: Cells, Singlets, Live, Leucocytes, T cells, B cells, NK, Daudi, mDC45
- Marqueurs: hPDL1, hPD1, hCD16, Granzyme B+
- Quadrant gates pour analyse multi-marqueurs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from scipy import ndimage
from pathlib import Path
import tempfile
import io
import json
import os
import flowio

st.set_page_config(page_title="FACS FlowJo Style v7", page_icon="🔬", layout="wide")

LEARNED_PARAMS_FILE = "learned_gating_params.json"

st.markdown("""
<style>
.main-header { font-size: 1.8rem; color: #1a5276; text-align: center; margin-bottom: 0.5rem; font-weight: bold; }
.flowjo-plot { border: 2px solid #2c3e50; border-radius: 8px; padding: 5px; background: white; }
.stat-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 8px; text-align: center; }
.marker-positive { color: #27ae60; font-weight: bold; }
.marker-negative { color: #e74c3c; }
.quadrant-stats { font-family: monospace; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ==================== APPRENTISSAGE ====================

def load_learned_params():
    if os.path.exists(LEARNED_PARAMS_FILE):
        try:
            with open(LEARNED_PARAMS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'version': 3, 'n_corrections': 0, 'gates': {}}


def save_learned_params(params):
    with open(LEARNED_PARAMS_FILE, 'w') as f:
        json.dump(params, f, indent=2)


def apply_learned_adj(polygon, gate_name):
    if polygon is None:
        return None
    params = load_learned_params()
    if gate_name not in params.get('gates', {}):
        return polygon
    gate = params['gates'][gate_name]
    adj = gate.get('avg_adjustment', {'x': 0, 'y': 0})
    scale = gate.get('scale_factor', 1.0)
    if abs(adj['x']) < 0.5 and abs(adj['y']) < 0.5 and abs(scale - 1.0) < 0.02:
        return polygon
    poly_arr = np.array(polygon)
    center = np.mean(poly_arr, axis=0)
    scaled = center + (poly_arr - center) * scale
    translated = scaled + np.array([adj['x'], adj['y']])
    return [(float(p[0]), float(p[1])) for p in translated]


# ==================== LECTURE FCS ====================

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
            pns = self.flow_data.text.get(f'$P{i}S', '') or self.flow_data.text.get(f'p{i}s', '')
            label = str(pns).strip() if pns else str(pnn).strip() if pnn else f'Ch{i}'
            labels.append(label)
        self.channels = labels
        self.data = pd.DataFrame(events, columns=labels)


def find_channel(columns, keywords):
    """Recherche flexible de canal"""
    for col in columns:
        for kw in keywords:
            if col.upper() == kw.upper():
                return col
    for col in columns:
        col_upper = col.upper().replace('-', '').replace('_', '').replace(' ', '')
        for kw in keywords:
            kw_clean = kw.upper().replace('-', '').replace('_', '').replace(' ', '')
            if kw_clean in col_upper:
                return col
    return None


# ==================== TRANSFORMATIONS ====================

def biex(x, width=150, scale=50):
    return np.arcsinh(np.asarray(x, float) / width) * scale


def logicle_approx(x, T=262144, M=4.5, W=0.5):
    """Approximation de la transformation Logicle (style FlowJo)"""
    x = np.asarray(x, float)
    w = W / (M + W)
    return np.where(x >= 0,
                    np.log10(1 + x / (T * w)) / (M + W) * M,
                    -np.log10(1 - x / (T * w)) / (M + W) * M)


# ==================== GEOMETRIE ====================

def create_hexagon(center_x, center_y, radius_x, radius_y):
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    return [(float(center_x + radius_x * np.cos(a)), float(center_y + radius_y * np.sin(a))) for a in angles]


def create_rectangle(x_min, x_max, y_min, y_max):
    """Crée un gate rectangulaire"""
    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]


def point_in_polygon(x, y, polygon):
    if polygon is None or len(polygon) < 3:
        return np.zeros(len(x), dtype=bool)
    n = len(polygon)
    px = np.array([p[0] for p in polygon], dtype=np.float64)
    py = np.array([p[1] for p in polygon], dtype=np.float64)
    inside = np.zeros(len(x), dtype=bool)
    j = n - 1
    for i in range(n):
        dy = py[j] - py[i]
        if abs(dy) < 1e-10:
            dy = 1e-10
        cond = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / dy + px[i])
        inside ^= cond
        j = i
    return inside


def apply_gate(data, x_ch, y_ch, polygon, parent_mask=None):
    if x_ch is None or y_ch is None or polygon is None or len(polygon) < 3:
        return pd.Series(False, index=data.index)
    x, y = data[x_ch].values, data[y_ch].values
    base = parent_mask.values.copy() if parent_mask is not None else np.ones(len(data), dtype=bool)
    valid = np.isfinite(x) & np.isfinite(y) & base
    if not valid.any():
        return pd.Series(False, index=data.index)
    xt, yt = biex(x), biex(y)
    in_poly = point_in_polygon(xt, yt, polygon)
    return pd.Series(valid & in_poly, index=data.index)


def apply_threshold_gate(data, channel, threshold, parent_mask=None, above=True):
    """Gate basé sur un seuil (pour marqueurs)"""
    if channel is None:
        return pd.Series(False, index=data.index)
    x = biex(data[channel].values)
    base = parent_mask.values if parent_mask is not None else np.ones(len(data), dtype=bool)
    valid = np.isfinite(x) & base
    if above:
        return pd.Series(valid & (x > threshold), index=data.index)
    else:
        return pd.Series(valid & (x <= threshold), index=data.index)


# ==================== QUADRANT GATES ====================

def calculate_quadrant_stats(data, x_ch, y_ch, x_threshold, y_threshold, parent_mask=None):
    """Calcule les statistiques des 4 quadrants (style FlowJo)"""
    if x_ch is None or y_ch is None:
        return None

    x = biex(data[x_ch].values)
    y = biex(data[y_ch].values)

    if parent_mask is not None:
        mask = parent_mask.values & np.isfinite(x) & np.isfinite(y)
    else:
        mask = np.isfinite(x) & np.isfinite(y)

    n_total = mask.sum()
    if n_total == 0:
        return None

    x_m, y_m = x[mask], y[mask]

    q1 = (x_m <= x_threshold) & (y_m > y_threshold)   # Upper Left
    q2 = (x_m > x_threshold) & (y_m > y_threshold)    # Upper Right
    q3 = (x_m <= x_threshold) & (y_m <= y_threshold)  # Lower Left
    q4 = (x_m > x_threshold) & (y_m <= y_threshold)   # Lower Right

    return {
        'UL': {'count': q1.sum(), 'pct': q1.sum() / n_total * 100},
        'UR': {'count': q2.sum(), 'pct': q2.sum() / n_total * 100},
        'LL': {'count': q3.sum(), 'pct': q3.sum() / n_total * 100},
        'LR': {'count': q4.sum(), 'pct': q4.sum() / n_total * 100},
        'total': n_total
    }


# ==================== AUTO-GATING ====================

def auto_gate_hexagon(data, x_ch, y_ch, parent_mask=None, mode='main'):
    if x_ch is None or y_ch is None:
        return None, 0.0
    x, y = data[x_ch].values, data[y_ch].values
    mask = (parent_mask.values if parent_mask is not None else np.ones(len(data), dtype=bool)) & \
           np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 100:
        return None, 0.0

    xt, yt = biex(x[mask]), biex(y[mask])
    X = np.column_stack([xt, yt])

    try:
        # Remove outliers
        if len(X) > 500:
            detector = EllipticEnvelope(contamination=0.03, random_state=42)
            inlier_mask = detector.fit_predict(X) == 1
            X_clean = X[inlier_mask] if inlier_mask.sum() > 100 else X
        else:
            X_clean = X

        # GMM
        best_gmm, best_score = None, -np.inf
        for n in [2, 3]:
            for init in range(3):
                try:
                    gmm = GaussianMixture(n_components=n, covariance_type='full',
                                          random_state=42+init, n_init=1, max_iter=200)
                    gmm.fit(X_clean)
                    score = gmm.bic(X_clean)
                    if best_gmm is None or score < best_score:
                        best_score, best_gmm = score, gmm
                except:
                    pass

        if best_gmm is None:
            return None, 0.0

        labels = best_gmm.predict(X_clean)
        n_comp = best_gmm.n_components

        cluster_stats = []
        for i in range(n_comp):
            cm = labels == i
            if cm.sum() > 0:
                cluster_stats.append({
                    'idx': i, 'count': cm.sum(),
                    'mean_x': np.mean(X_clean[cm, 0]),
                    'mean_y': np.mean(X_clean[cm, 1])
                })

        if not cluster_stats:
            return None, 0.0

        if mode == 'main':
            target = max(cluster_stats, key=lambda x: x['count'])['idx']
        elif mode == 'low_x':
            target = min(cluster_stats, key=lambda x: x['mean_x'])['idx']
        elif mode == 'high_x':
            target = max(cluster_stats, key=lambda x: x['mean_x'])['idx']
        elif mode == 'high_y':
            target = max(cluster_stats, key=lambda x: x['mean_y'])['idx']
        else:
            target = 0

        cm = labels == target
        cx, cy = X_clean[cm, 0], X_clean[cm, 1]
        if len(cx) < 50:
            return None, 0.0

        center_x, center_y = np.median(cx), np.median(cy)
        radius_x = max(np.percentile(np.abs(cx - center_x), 90) * 1.2, 8)
        radius_y = max(np.percentile(np.abs(cy - center_y), 90) * 1.2, 8)

        polygon = create_hexagon(center_x, center_y, radius_x, radius_y)
        return polygon, 75.0
    except:
        return None, 0.0


def auto_find_threshold(data, channel, parent_mask=None, method='bimodal'):
    """Trouve automatiquement un seuil pour un marqueur"""
    if channel is None:
        return 0.0

    x = data[channel].values
    if parent_mask is not None:
        x = x[parent_mask.values & np.isfinite(x) & (x > 0)]
    else:
        x = x[np.isfinite(x) & (x > 0)]

    if len(x) < 100:
        return 0.0

    xt = biex(x)

    try:
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(xt.reshape(-1, 1))
        means = gmm.means_.flatten()
        threshold = np.mean(means)
        return float(threshold)
    except:
        return float(np.percentile(xt, 50))


# ==================== VISUALISATION FLOWJO ====================

def create_flowjo_plot(data, x_ch, y_ch, x_label, y_label, title, polygon=None,
                       parent_mask=None, gate_name="", plot_type='pseudo',
                       quadrant_lines=None, show_stats=True):
    """
    Crée un plot style FlowJo
    plot_type: 'pseudo' (pseudo-color), 'contour', 'dot', 'density'
    """
    if x_ch is None or y_ch is None:
        fig = go.Figure()
        fig.add_annotation(text="Canal non trouvé", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=350)
        return fig, 0, 0.0, None

    x, y = data[x_ch].values, data[y_ch].values
    mask = (parent_mask.values if parent_mask is not None else np.ones(len(data), dtype=bool)) & \
           np.isfinite(x) & np.isfinite(y)

    n_parent = mask.sum()
    if n_parent == 0:
        fig = go.Figure()
        fig.add_annotation(text="Pas de données", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=350)
        return fig, 0, 0.0, None

    xt, yt = biex(x[mask]), biex(y[mask])

    # Sous-échantillonnage
    n_display = min(15000, len(xt))
    if len(xt) > n_display:
        idx = np.random.choice(len(xt), n_display, replace=False)
        xd, yd = xt[idx], yt[idx]
    else:
        xd, yd = xt, yt

    fig = go.Figure()

    if plot_type == 'pseudo':
        # Pseudo-color density (FlowJo style)
        fig.add_trace(go.Histogram2d(
            x=xd, y=yd,
            colorscale='Hot',
            reversescale=True,
            nbinsx=150, nbinsy=150,
            showscale=False,
            zsmooth='best'
        ))
        # Overlay scatter for low-density regions
        fig.add_trace(go.Scattergl(
            x=xd, y=yd, mode='markers',
            marker=dict(size=1, color='blue', opacity=0.1),
            hoverinfo='skip'
        ))

    elif plot_type == 'contour':
        # Contour plot (FlowJo style)
        fig.add_trace(go.Histogram2dContour(
            x=xd, y=yd,
            colorscale='Blues',
            reversescale=False,
            showscale=False,
            contours=dict(coloring='heatmap', showlines=True),
            ncontours=15
        ))

    elif plot_type == 'density':
        # Density avec KDE
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([xd, yd])
            kde = gaussian_kde(xy)
            colors = kde(xy)
            fig.add_trace(go.Scattergl(
                x=xd, y=yd, mode='markers',
                marker=dict(size=2, color=colors, colorscale='Viridis', opacity=0.6),
                hoverinfo='skip'
            ))
        except:
            fig.add_trace(go.Scattergl(
                x=xd, y=yd, mode='markers',
                marker=dict(size=2, color='blue', opacity=0.3),
                hoverinfo='skip'
            ))

    else:  # dot
        fig.add_trace(go.Scattergl(
            x=xd, y=yd, mode='markers',
            marker=dict(size=2, color='#1f77b4', opacity=0.4),
            hoverinfo='skip'
        ))

    n_in, pct = 0, 0.0

    # Polygon gate
    if polygon and len(polygon) >= 3:
        gate_mask = apply_gate(data, x_ch, y_ch, polygon, parent_mask)
        n_in = int(gate_mask.sum())
        pct = float(n_in / n_parent * 100) if n_parent > 0 else 0.0

        px = [p[0] for p in polygon] + [polygon[0][0]]
        py = [p[1] for p in polygon] + [polygon[0][1]]

        fig.add_trace(go.Scatter(
            x=px, y=py, fill='toself',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(color='#00ff00', width=2),
            mode='lines', name='Gate'
        ))

        # Gate annotation
        cx_p, cy_p = np.mean([p[0] for p in polygon]), np.mean([p[1] for p in polygon])
        fig.add_annotation(
            x=cx_p, y=cy_p,
            text=f"<b>{gate_name}</b><br>{pct:.1f}%",
            showarrow=False, font=dict(size=11, color='white'),
            bgcolor='rgba(0,0,0,0.7)', borderpad=4
        )

    # Quadrant lines
    quadrant_stats = None
    if quadrant_lines:
        x_thresh, y_thresh = quadrant_lines

        # Vertical line
        fig.add_vline(x=x_thresh, line=dict(color='black', width=1.5, dash='dash'))
        # Horizontal line
        fig.add_hline(y=y_thresh, line=dict(color='black', width=1.5, dash='dash'))

        # Calculate quadrant stats
        quadrant_stats = calculate_quadrant_stats(data, x_ch, y_ch, x_thresh, y_thresh, parent_mask)

        if quadrant_stats and show_stats:
            # Add quadrant labels
            x_range = [min(xd), max(xd)]
            y_range = [min(yd), max(yd)]

            positions = {
                'UL': (x_range[0] + (x_thresh - x_range[0]) * 0.5, y_thresh + (y_range[1] - y_thresh) * 0.8),
                'UR': (x_thresh + (x_range[1] - x_thresh) * 0.5, y_thresh + (y_range[1] - y_thresh) * 0.8),
                'LL': (x_range[0] + (x_thresh - x_range[0]) * 0.5, y_range[0] + (y_thresh - y_range[0]) * 0.2),
                'LR': (x_thresh + (x_range[1] - x_thresh) * 0.5, y_range[0] + (y_thresh - y_range[0]) * 0.2),
            }

            for quad, pos in positions.items():
                fig.add_annotation(
                    x=pos[0], y=pos[1],
                    text=f"<b>{quadrant_stats[quad]['pct']:.1f}%</b>",
                    showarrow=False,
                    font=dict(size=12, color='black'),
                    bgcolor='rgba(255,255,255,0.8)'
                )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b> (n={n_parent:,})", x=0.5, font=dict(size=12)),
        xaxis_title=x_label, yaxis_title=y_label,
        showlegend=False, height=350,
        margin=dict(l=50, r=20, t=45, b=45),
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#eee', zeroline=False, showline=True, linecolor='black'),
        yaxis=dict(showgrid=True, gridcolor='#eee', zeroline=False, showline=True, linecolor='black'),
    )

    return fig, n_in, pct, quadrant_stats


# ==================== CALCUL MARQUEURS ====================

def calculate_marker_stats(data, channel, parent_mask=None, threshold=None):
    """Calcule les statistiques d'un marqueur (MFI, % positif, etc.)"""
    if channel is None:
        return None

    x = data[channel].values
    if parent_mask is not None:
        mask = parent_mask.values & np.isfinite(x)
    else:
        mask = np.isfinite(x)

    if mask.sum() < 10:
        return None

    x_valid = x[mask]
    xt = biex(x_valid)

    if threshold is None:
        threshold = auto_find_threshold(data, channel, parent_mask)

    positive = xt > threshold

    return {
        'channel': channel,
        'n_total': len(x_valid),
        'n_positive': positive.sum(),
        'pct_positive': positive.sum() / len(x_valid) * 100,
        'mfi_all': float(np.median(x_valid)),
        'mfi_positive': float(np.median(x_valid[positive])) if positive.sum() > 0 else 0,
        'mfi_negative': float(np.median(x_valid[~positive])) if (~positive).sum() > 0 else 0,
        'threshold': threshold,
        'cv': float(np.std(x_valid) / np.mean(x_valid) * 100) if np.mean(x_valid) > 0 else 0
    }


# ==================== MAIN APPLICATION ====================

st.markdown('<h1 class="main-header">🔬 FACS Analysis - FlowJo Style v7</h1>', unsafe_allow_html=True)

# Session state
for key in ['reader', 'data', 'channels', 'polygons', 'masks', 'auto_done', 'thresholds']:
    if key not in st.session_state:
        st.session_state[key] = {} if key in ['channels', 'polygons', 'masks', 'thresholds'] else (False if key == 'auto_done' else None)

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
            st.session_state.masks = {}
            st.session_state.thresholds = {}
            st.session_state.auto_done = False

            cols = list(reader.data.columns)
            st.session_state.channels = {
                # Scatter
                'FSC-A': find_channel(cols, ['FSC-A', 'FSC_A']),
                'FSC-H': find_channel(cols, ['FSC-H', 'FSC_H']),
                'SSC-A': find_channel(cols, ['SSC-A', 'SSC_A']),
                # Viabilité
                'LiveDead': find_channel(cols, ['LiveDead', 'Viab', 'Aqua', 'Live', 'L/D', 'LD', 'Zombie']),
                # Leucocytes
                'hCD45': find_channel(cols, ['hCD45', 'CD45', 'PerCP']),
                'mCD45': find_channel(cols, ['mCD45', 'mCD45.1']),
                # Lymphocytes T
                'CD3': find_channel(cols, ['CD3', 'AF488', 'FITC']),
                'CD4': find_channel(cols, ['CD4', 'BV650']),
                'CD8': find_channel(cols, ['CD8', 'BUV805', 'APC']),
                # Lymphocytes B
                'CD19': find_channel(cols, ['CD19', 'PE-Fire', 'PE-CF594']),
                # NK
                'CD56': find_channel(cols, ['CD56', 'BV421', 'Pacific Blue']),
                'CD16': find_channel(cols, ['CD16', 'hCD16', 'BV785']),
                # Cellules cibles
                'Daudi': find_channel(cols, ['Daudi', 'CellTrace', 'CFSE', 'Violet']),
                # Marqueurs fonctionnels
                'hPDL1': find_channel(cols, ['hPDL1', 'PDL1', 'PD-L1', 'CD274']),
                'hPD1': find_channel(cols, ['hPD1', 'PD1', 'PD-1', 'CD279']),
                'GranzymeB': find_channel(cols, ['GranzymeB', 'Granzyme', 'GrzB', 'GrB', 'GZMB']),
            }

    reader = st.session_state.reader
    data = st.session_state.data
    ch = st.session_state.channels
    n_total = len(data)

    # Métriques
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Événements", f"{n_total:,}")
    c2.metric("Canaux", len(reader.channels))
    c3.metric("Fichier", reader.filename[:20])

    # Options de visualisation
    with st.sidebar:
        st.markdown("### ⚙️ Options")
        plot_type = st.selectbox("Style de plot", ['pseudo', 'contour', 'density', 'dot'],
                                  format_func=lambda x: {'pseudo': '🎨 Pseudo-color', 'contour': '📊 Contour',
                                                         'density': '🌡️ Densité', 'dot': '⚫ Dot plot'}[x])
        show_quadrants = st.checkbox("Afficher quadrants", value=True)

        st.markdown("### 📋 Canaux détectés")
        for name, canal in ch.items():
            if canal:
                st.write(f"✅ {name}: {canal[:15]}")
            else:
                st.write(f"❌ {name}")

    if ch['FSC-A'] is None or ch['SSC-A'] is None:
        st.error("❌ FSC-A ou SSC-A non trouvé!")
        st.stop()

    st.markdown("---")

    # AUTO-GATING
    if not st.session_state.auto_done:
        if st.button("🚀 LANCER L'ANALYSE COMPLÈTE", type="primary", use_container_width=True):
            progress = st.progress(0, "Initialisation...")
            polygons = st.session_state.polygons
            masks = st.session_state.masks
            thresholds = st.session_state.thresholds

            # 1. Cells
            progress.progress(5, "Cells...")
            poly, _ = auto_gate_hexagon(data, ch['FSC-A'], ch['SSC-A'], None, 'main')
            polygons['cells'] = apply_learned_adj(poly, 'cells')
            masks['cells'] = apply_gate(data, ch['FSC-A'], ch['SSC-A'], polygons['cells'], None)

            # 2. Singlets
            progress.progress(10, "Singlets...")
            if ch['FSC-H']:
                poly, _ = auto_gate_hexagon(data, ch['FSC-A'], ch['FSC-H'], masks['cells'], 'main')
                polygons['singlets'] = apply_learned_adj(poly, 'singlets')
                masks['singlets'] = apply_gate(data, ch['FSC-A'], ch['FSC-H'], polygons['singlets'], masks['cells'])
            else:
                masks['singlets'] = masks['cells']

            # 3. Live
            progress.progress(15, "Live...")
            if ch['LiveDead']:
                poly, _ = auto_gate_hexagon(data, ch['LiveDead'], ch['SSC-A'], masks['singlets'], 'low_x')
                polygons['live'] = apply_learned_adj(poly, 'live')
                masks['live'] = apply_gate(data, ch['LiveDead'], ch['SSC-A'], polygons['live'], masks['singlets'])
            else:
                masks['live'] = masks['singlets']

            # 4. Leucocytes (hCD45+)
            progress.progress(25, "Leucocytes...")
            if ch['hCD45']:
                poly, _ = auto_gate_hexagon(data, ch['hCD45'], ch['SSC-A'], masks['live'], 'high_x')
                polygons['leucocytes'] = apply_learned_adj(poly, 'leucocytes')
                masks['leucocytes'] = apply_gate(data, ch['hCD45'], ch['SSC-A'], polygons['leucocytes'], masks['live'])
            else:
                masks['leucocytes'] = masks['live']

            # 5. T cells (CD3+)
            progress.progress(35, "T cells...")
            if ch['CD3']:
                poly, _ = auto_gate_hexagon(data, ch['CD3'], ch['SSC-A'], masks['leucocytes'], 'high_x')
                polygons['tcells'] = apply_learned_adj(poly, 'tcells')
                masks['tcells'] = apply_gate(data, ch['CD3'], ch['SSC-A'], polygons['tcells'], masks['leucocytes'])
            else:
                masks['tcells'] = masks['leucocytes']

            # 6. B cells (CD19+)
            progress.progress(45, "B cells...")
            if ch['CD19']:
                poly, _ = auto_gate_hexagon(data, ch['CD19'], ch['SSC-A'], masks['leucocytes'], 'high_x')
                polygons['bcells'] = apply_learned_adj(poly, 'bcells')
                masks['bcells'] = apply_gate(data, ch['CD19'], ch['SSC-A'], polygons['bcells'], masks['leucocytes'])

            # 7. NK cells (CD56+ ou CD16+)
            progress.progress(55, "NK cells...")
            nk_ch = ch['CD56'] or ch['CD16']
            if nk_ch:
                poly, _ = auto_gate_hexagon(data, nk_ch, ch['SSC-A'], masks['leucocytes'], 'high_x')
                polygons['nk'] = apply_learned_adj(poly, 'nk')
                masks['nk'] = apply_gate(data, nk_ch, ch['SSC-A'], polygons['nk'], masks['leucocytes'])

            # 8. CD4+ T cells
            progress.progress(65, "CD4+ T cells...")
            if ch['CD4']:
                poly, _ = auto_gate_hexagon(data, ch['CD4'], ch['SSC-A'], masks['tcells'], 'high_x')
                polygons['cd4'] = apply_learned_adj(poly, 'cd4')
                masks['cd4'] = apply_gate(data, ch['CD4'], ch['SSC-A'], polygons['cd4'], masks['tcells'])

            # 9. CD8+ T cells
            progress.progress(70, "CD8+ T cells...")
            if ch['CD8']:
                poly, _ = auto_gate_hexagon(data, ch['CD8'], ch['SSC-A'], masks['tcells'], 'high_x')
                polygons['cd8'] = apply_learned_adj(poly, 'cd8')
                masks['cd8'] = apply_gate(data, ch['CD8'], ch['SSC-A'], polygons['cd8'], masks['tcells'])

            # 10. Daudi (cellules cibles)
            progress.progress(75, "Daudi...")
            if ch['Daudi']:
                poly, _ = auto_gate_hexagon(data, ch['Daudi'], ch['SSC-A'], masks['live'], 'high_x')
                polygons['daudi'] = apply_learned_adj(poly, 'daudi')
                masks['daudi'] = apply_gate(data, ch['Daudi'], ch['SSC-A'], polygons['daudi'], masks['live'])

            # Thresholds pour marqueurs
            progress.progress(85, "Calcul seuils marqueurs...")
            if ch['hPDL1']:
                thresholds['hPDL1'] = auto_find_threshold(data, ch['hPDL1'], masks.get('leucocytes'))
            if ch['hPD1']:
                thresholds['hPD1'] = auto_find_threshold(data, ch['hPD1'], masks.get('tcells'))
            if ch['CD16']:
                thresholds['hCD16'] = auto_find_threshold(data, ch['CD16'], masks.get('nk') or masks.get('leucocytes'))
            if ch['GranzymeB']:
                thresholds['GranzymeB'] = auto_find_threshold(data, ch['GranzymeB'], masks.get('nk') or masks.get('tcells'))

            progress.progress(100, "Terminé!")
            st.session_state.auto_done = True
            st.rerun()

    # AFFICHAGE
    if st.session_state.auto_done:
        polygons = st.session_state.polygons
        masks = st.session_state.masks
        thresholds = st.session_state.thresholds

        # Recalcul des masques
        masks['cells'] = apply_gate(data, ch['FSC-A'], ch['SSC-A'], polygons.get('cells'), None)
        masks['singlets'] = apply_gate(data, ch['FSC-A'], ch['FSC-H'], polygons.get('singlets'), masks['cells']) if ch['FSC-H'] else masks['cells']
        masks['live'] = apply_gate(data, ch['LiveDead'], ch['SSC-A'], polygons.get('live'), masks['singlets']) if ch['LiveDead'] else masks['singlets']
        masks['leucocytes'] = apply_gate(data, ch['hCD45'], ch['SSC-A'], polygons.get('leucocytes'), masks['live']) if ch['hCD45'] else masks['live']
        masks['tcells'] = apply_gate(data, ch['CD3'], ch['SSC-A'], polygons.get('tcells'), masks['leucocytes']) if ch['CD3'] else masks['leucocytes']

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Hiérarchie", "🧬 Populations", "💊 Marqueurs", "📥 Export"])

        with tab1:
            st.markdown("### 🌳 Hiérarchie de Gating")

            # Première ligne: Cells, Singlets, Live
            cols = st.columns(3)

            with cols[0]:
                fig, n, pct, _ = create_flowjo_plot(
                    data, ch['FSC-A'], ch['SSC-A'], 'FSC-A', 'SSC-A',
                    'Cells', polygons.get('cells'), None, 'Cells', plot_type
                )
                st.plotly_chart(fig, use_container_width=True, key='p_cells')

            with cols[1]:
                fig, n, pct, _ = create_flowjo_plot(
                    data, ch['FSC-A'], ch['FSC-H'], 'FSC-A', 'FSC-H',
                    'Singlets', polygons.get('singlets'), masks['cells'], 'Singlets', plot_type
                )
                st.plotly_chart(fig, use_container_width=True, key='p_singlets')

            with cols[2]:
                fig, n, pct, _ = create_flowjo_plot(
                    data, ch['LiveDead'], ch['SSC-A'], 'Live/Dead', 'SSC-A',
                    'Live', polygons.get('live'), masks['singlets'], 'Live', plot_type
                )
                st.plotly_chart(fig, use_container_width=True, key='p_live')

            # Deuxième ligne: Leucocytes, T cells, B cells
            cols2 = st.columns(3)

            with cols2[0]:
                fig, n, pct, _ = create_flowjo_plot(
                    data, ch['hCD45'], ch['SSC-A'], 'hCD45', 'SSC-A',
                    'Leucocytes (hCD45+)', polygons.get('leucocytes'), masks['live'], 'Leucocytes', plot_type
                )
                st.plotly_chart(fig, use_container_width=True, key='p_leuco')

            with cols2[1]:
                fig, n, pct, _ = create_flowjo_plot(
                    data, ch['CD3'], ch['SSC-A'], 'CD3', 'SSC-A',
                    'T cells (CD3+)', polygons.get('tcells'), masks['leucocytes'], 'T cells', plot_type
                )
                st.plotly_chart(fig, use_container_width=True, key='p_tcells')

            with cols2[2]:
                fig, n, pct, _ = create_flowjo_plot(
                    data, ch['CD19'], ch['SSC-A'], 'CD19', 'SSC-A',
                    'B cells (CD19+)', polygons.get('bcells'), masks['leucocytes'], 'B cells', plot_type
                )
                st.plotly_chart(fig, use_container_width=True, key='p_bcells')

        with tab2:
            st.markdown("### 🧬 Sous-populations")

            cols = st.columns(3)

            # NK cells
            with cols[0]:
                nk_ch = ch['CD56'] or ch['CD16']
                if nk_ch:
                    fig, n, pct, _ = create_flowjo_plot(
                        data, nk_ch, ch['SSC-A'], 'CD56/CD16', 'SSC-A',
                        'NK cells', polygons.get('nk'), masks['leucocytes'], 'NK', plot_type
                    )
                    st.plotly_chart(fig, use_container_width=True, key='p_nk')

            # CD4+ T cells
            with cols[1]:
                if ch['CD4']:
                    fig, n, pct, _ = create_flowjo_plot(
                        data, ch['CD4'], ch['SSC-A'], 'CD4', 'SSC-A',
                        'CD4+ T cells', polygons.get('cd4'), masks['tcells'], 'CD4+', plot_type
                    )
                    st.plotly_chart(fig, use_container_width=True, key='p_cd4')

            # CD8+ T cells
            with cols[2]:
                if ch['CD8']:
                    fig, n, pct, _ = create_flowjo_plot(
                        data, ch['CD8'], ch['SSC-A'], 'CD8', 'SSC-A',
                        'CD8+ T cells', polygons.get('cd8'), masks['tcells'], 'CD8+', plot_type
                    )
                    st.plotly_chart(fig, use_container_width=True, key='p_cd8')

            # Daudi et mCD45
            cols2 = st.columns(3)

            with cols2[0]:
                if ch['Daudi']:
                    fig, n, pct, _ = create_flowjo_plot(
                        data, ch['Daudi'], ch['SSC-A'], 'Daudi', 'SSC-A',
                        'Daudi (cibles)', polygons.get('daudi'), masks['live'], 'Daudi', plot_type
                    )
                    st.plotly_chart(fig, use_container_width=True, key='p_daudi')

            with cols2[1]:
                if ch['mCD45']:
                    fig, n, pct, _ = create_flowjo_plot(
                        data, ch['mCD45'], ch['SSC-A'], 'mCD45', 'SSC-A',
                        'mCD45+', None, masks['live'], 'mCD45', plot_type
                    )
                    st.plotly_chart(fig, use_container_width=True, key='p_mcd45')

            # CD4 vs CD8 avec quadrants
            with cols2[2]:
                if ch['CD4'] and ch['CD8']:
                    cd4_thresh = thresholds.get('CD4', auto_find_threshold(data, ch['CD4'], masks['tcells']))
                    cd8_thresh = thresholds.get('CD8', auto_find_threshold(data, ch['CD8'], masks['tcells']))

                    quad_lines = (cd4_thresh, cd8_thresh) if show_quadrants else None
                    fig, n, pct, qstats = create_flowjo_plot(
                        data, ch['CD4'], ch['CD8'], 'CD4', 'CD8',
                        'CD4 vs CD8', None, masks['tcells'], '', plot_type, quad_lines
                    )
                    st.plotly_chart(fig, use_container_width=True, key='p_cd4cd8')

        with tab3:
            st.markdown("### 💊 Analyse des Marqueurs")

            marker_results = []

            # hPDL1
            if ch['hPDL1']:
                stats = calculate_marker_stats(data, ch['hPDL1'], masks.get('leucocytes'), thresholds.get('hPDL1'))
                if stats:
                    marker_results.append({
                        'Marqueur': 'hPDL1',
                        'Population': 'Leucocytes',
                        '% Positif': f"{stats['pct_positive']:.1f}%",
                        'MFI (total)': f"{stats['mfi_all']:.0f}",
                        'MFI (pos)': f"{stats['mfi_positive']:.0f}",
                        'N positifs': stats['n_positive']
                    })

            # hPD1
            if ch['hPD1']:
                stats = calculate_marker_stats(data, ch['hPD1'], masks.get('tcells'), thresholds.get('hPD1'))
                if stats:
                    marker_results.append({
                        'Marqueur': 'hPD1',
                        'Population': 'T cells',
                        '% Positif': f"{stats['pct_positive']:.1f}%",
                        'MFI (total)': f"{stats['mfi_all']:.0f}",
                        'MFI (pos)': f"{stats['mfi_positive']:.0f}",
                        'N positifs': stats['n_positive']
                    })

            # hCD16
            if ch['CD16']:
                parent = masks.get('nk') or masks.get('leucocytes')
                stats = calculate_marker_stats(data, ch['CD16'], parent, thresholds.get('hCD16'))
                if stats:
                    marker_results.append({
                        'Marqueur': 'hCD16',
                        'Population': 'NK/Leucocytes',
                        '% Positif': f"{stats['pct_positive']:.1f}%",
                        'MFI (total)': f"{stats['mfi_all']:.0f}",
                        'MFI (pos)': f"{stats['mfi_positive']:.0f}",
                        'N positifs': stats['n_positive']
                    })

            # Granzyme B
            if ch['GranzymeB']:
                parent = masks.get('nk') or masks.get('tcells') or masks.get('leucocytes')
                stats = calculate_marker_stats(data, ch['GranzymeB'], parent, thresholds.get('GranzymeB'))
                if stats:
                    marker_results.append({
                        'Marqueur': 'Granzyme B+',
                        'Population': 'NK/T cells',
                        '% Positif': f"{stats['pct_positive']:.1f}%",
                        'MFI (total)': f"{stats['mfi_all']:.0f}",
                        'MFI (pos)': f"{stats['mfi_positive']:.0f}",
                        'N positifs': stats['n_positive']
                    })

            if marker_results:
                df_markers = pd.DataFrame(marker_results)
                st.dataframe(df_markers, use_container_width=True, hide_index=True)

                # Visualisation des marqueurs avec quadrants
                st.markdown("#### 📊 Quadrant Analysis")

                cols = st.columns(2)

                # PD1 vs PDL1
                if ch['hPD1'] and ch['hPDL1']:
                    with cols[0]:
                        pd1_thresh = thresholds.get('hPD1', 0)
                        pdl1_thresh = thresholds.get('hPDL1', 0)
                        quad_lines = (pd1_thresh, pdl1_thresh) if show_quadrants else None

                        fig, _, _, qstats = create_flowjo_plot(
                            data, ch['hPD1'], ch['hPDL1'], 'hPD1', 'hPDL1',
                            'PD1 vs PDL1', None, masks.get('tcells'), '', plot_type, quad_lines
                        )
                        st.plotly_chart(fig, use_container_width=True, key='p_pd1pdl1')

                        if qstats:
                            st.markdown(f"""
                            **Quadrants:**
                            - PD1+ PDL1+: {qstats['UR']['pct']:.1f}%
                            - PD1+ PDL1-: {qstats['LR']['pct']:.1f}%
                            - PD1- PDL1+: {qstats['UL']['pct']:.1f}%
                            - PD1- PDL1-: {qstats['LL']['pct']:.1f}%
                            """)

                # CD16 vs Granzyme B
                if ch['CD16'] and ch['GranzymeB']:
                    with cols[1]:
                        cd16_thresh = thresholds.get('hCD16', 0)
                        grzb_thresh = thresholds.get('GranzymeB', 0)
                        quad_lines = (cd16_thresh, grzb_thresh) if show_quadrants else None

                        fig, _, _, qstats = create_flowjo_plot(
                            data, ch['CD16'], ch['GranzymeB'], 'CD16', 'Granzyme B',
                            'CD16 vs Granzyme B', None, masks.get('nk') or masks.get('leucocytes'), '', plot_type, quad_lines
                        )
                        st.plotly_chart(fig, use_container_width=True, key='p_cd16grzb')

                        if qstats:
                            st.markdown(f"""
                            **Quadrants:**
                            - CD16+ GrzB+: {qstats['UR']['pct']:.1f}%
                            - CD16+ GrzB-: {qstats['LR']['pct']:.1f}%
                            - CD16- GrzB+: {qstats['UL']['pct']:.1f}%
                            - CD16- GrzB-: {qstats['LL']['pct']:.1f}%
                            """)
            else:
                st.info("Aucun marqueur fonctionnel détecté dans les canaux")

        with tab4:
            st.markdown("### 📥 Export")

            # Statistiques des populations
            pop_stats = []
            pop_names = ['cells', 'singlets', 'live', 'leucocytes', 'tcells', 'bcells', 'nk', 'cd4', 'cd8', 'daudi']
            pop_labels = ['Cells', 'Singlets', 'Live', 'Leucocytes (hCD45+)', 'T cells (CD3+)',
                         'B cells (CD19+)', 'NK cells', 'CD4+ T', 'CD8+ T', 'Daudi']

            for pname, plabel in zip(pop_names, pop_labels):
                if pname in masks:
                    n = masks[pname].sum()
                    pct_total = n / n_total * 100
                    pop_stats.append({
                        'Population': plabel,
                        'Count': n,
                        '% Total': f"{pct_total:.2f}%"
                    })

            if pop_stats:
                df_pop = pd.DataFrame(pop_stats)
                st.dataframe(df_pop, use_container_width=True, hide_index=True)

                col1, col2 = st.columns(2)
                col1.download_button(
                    "📥 Populations CSV",
                    df_pop.to_csv(index=False),
                    f"{reader.filename}_populations.csv",
                    "text/csv", use_container_width=True
                )

                if marker_results:
                    df_markers = pd.DataFrame(marker_results)
                    col2.download_button(
                        "📥 Marqueurs CSV",
                        df_markers.to_csv(index=False),
                        f"{reader.filename}_markers.csv",
                        "text/csv", use_container_width=True
                    )

else:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 10px; text-align: center;">
    <h2>🔬 FACS FlowJo Style v7</h2>
    <p><b>Populations:</b> Cells, Singlets, Live, Leucocytes, T cells, B cells, NK, CD4+, CD8+, Daudi, mCD45</p>
    <p><b>Marqueurs:</b> hPDL1, hPD1, hCD16, Granzyme B+</p>
    <p><b>Visualisation:</b> Pseudo-color, Contour, Densité, Quadrants</p>
    <br>
    <p>📁 Uploadez un fichier FCS pour commencer</p>
    </div>
    """, unsafe_allow_html=True)

st.caption("v7 | FlowJo Style | Complete Immune Phenotyping")
