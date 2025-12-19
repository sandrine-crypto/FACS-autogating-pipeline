#!/usr/bin/env python3
"""
FACS Autogating - Gates Hexagonaux v6
Nouvelles fonctionnalités:
- Tous les gates (Cells, Singlets, Live, hCD45, CD3, CD4, CD8, CD19)
- Export des graphiques (PNG, PDF, SVG)
- Vue récapitulative de tous les gates
- Hiérarchie complète du gating
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from pathlib import Path
import tempfile
import io
import json
import os
import flowio
import base64

st.set_page_config(page_title="FACS - Complete Gating v6", page_icon="🔬", layout="wide")

LEARNED_PARAMS_FILE = "learned_gating_params.json"

st.markdown("""
<style>
.main-header { font-size: 1.8rem; color: #2c3e50; text-align: center; margin-bottom: 0.5rem; }
.info-box { background: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid #0066cc; margin: 0.5rem 0; }
.confidence-high { color: #28a745; font-weight: bold; }
.confidence-medium { color: #ffc107; font-weight: bold; }
.confidence-low { color: #dc3545; font-weight: bold; }
.gate-section { background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
.export-section { background: #fff3cd; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ==================== APPRENTISSAGE ====================

def load_learned_params():
    if os.path.exists(LEARNED_PARAMS_FILE):
        try:
            with open(LEARNED_PARAMS_FILE, 'r') as f:
                params = json.load(f)
                if 'version' not in params:
                    params['version'] = 2
                    for gate in params.get('gates', {}).values():
                        gate.setdefault('scale_factor', 1.0)
                        gate.setdefault('rotation', 0.0)
                        gate.setdefault('confidence_history', [])
                return params
        except:
            pass
    return {'version': 2, 'n_corrections': 0, 'gates': {}}


def save_learned_params(params):
    if os.path.exists(LEARNED_PARAMS_FILE):
        try:
            os.rename(LEARNED_PARAMS_FILE, LEARNED_PARAMS_FILE + '.bak')
        except:
            pass
    with open(LEARNED_PARAMS_FILE, 'w') as f:
        json.dump(params, f, indent=2)


def update_learned_params(gate_name, original_polygon, corrected_polygon, confidence=None):
    params = load_learned_params()
    if gate_name not in params['gates']:
        params['gates'][gate_name] = {
            'avg_adjustment': {'x': 0, 'y': 0},
            'scale_factor': 1.0,
            'n_samples': 0,
            'confidence_history': []
        }
    gate_params = params['gates'][gate_name]
    orig_arr = np.array(original_polygon)
    corr_arr = np.array(corrected_polygon)
    orig_center = np.mean(orig_arr, axis=0)
    corr_center = np.mean(corr_arr, axis=0)
    dx = float(corr_center[0] - orig_center[0])
    dy = float(corr_center[1] - orig_center[1])
    orig_dists = np.linalg.norm(orig_arr - orig_center, axis=1)
    corr_dists = np.linalg.norm(corr_arr - corr_center, axis=1)
    scale = float(np.mean(corr_dists) / (np.mean(orig_dists) + 1e-10))
    gate_params['n_samples'] += 1
    n = gate_params['n_samples']
    alpha = 2 / (n + 1)
    gate_params['avg_adjustment']['x'] = (1 - alpha) * gate_params['avg_adjustment']['x'] + alpha * dx
    gate_params['avg_adjustment']['y'] = (1 - alpha) * gate_params['avg_adjustment']['y'] + alpha * dy
    gate_params['scale_factor'] = (1 - alpha) * gate_params['scale_factor'] + alpha * scale
    if confidence is not None:
        gate_params['confidence_history'].append(float(confidence))
        gate_params['confidence_history'] = gate_params['confidence_history'][-20:]
    params['n_corrections'] += 1
    save_learned_params(params)


def apply_learned_adj(polygon, gate_name):
    if polygon is None:
        return None
    params = load_learned_params()
    if gate_name not in params['gates']:
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
            labels.append(str(pnn).strip() if pnn else f'Ch{i}')
        self.channels = labels
        self.data = pd.DataFrame(events, columns=labels)


def find_channel(columns, keywords):
    for col in columns:
        for kw in keywords:
            if col.upper() == kw.upper():
                return col
    for col in columns:
        col_upper = col.upper()
        for kw in keywords:
            if kw.upper() in col_upper:
                return col
    return None


# ==================== TRANSFORMATIONS ====================

def biex(x, width=150, scale=50):
    return np.arcsinh(np.asarray(x, float) / width) * scale


# ==================== GEOMETRIE ====================

def create_hexagon(center_x, center_y, radius_x, radius_y):
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    return [(float(center_x + radius_x * np.cos(a)), float(center_y + radius_y * np.sin(a))) for a in angles]


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
    x = data[x_ch].values
    y = data[y_ch].values
    base = parent_mask.values.copy() if parent_mask is not None else np.ones(len(data), dtype=bool)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & base
    if not valid.any():
        return pd.Series(False, index=data.index)
    xt, yt = biex(x), biex(y)
    in_poly = point_in_polygon(xt, yt, polygon)
    return pd.Series(valid & in_poly, index=data.index)


def move_polygon(polygon, dx, dy):
    if polygon is None:
        return None
    return [(p[0] + dx, p[1] + dy) for p in polygon]


def scale_polygon(polygon, factor):
    if polygon is None:
        return None
    center = np.mean(polygon, axis=0)
    return [(float(center[0] + factor * (p[0] - center[0])),
             float(center[1] + factor * (p[1] - center[1]))) for p in polygon]


def rotate_polygon(polygon, angle_deg):
    if polygon is None:
        return None
    angle_rad = np.radians(angle_deg)
    center = np.mean(polygon, axis=0)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return [(float(center[0] + (p[0] - center[0]) * cos_a - (p[1] - center[1]) * sin_a),
             float(center[1] + (p[0] - center[0]) * sin_a + (p[1] - center[1]) * cos_a)) for p in polygon]


# ==================== AUTO-GATING ====================

def compute_gate_confidence(data, polygon, x_ch, y_ch, parent_mask=None):
    if polygon is None or x_ch is None or y_ch is None:
        return 0.0
    x, y = data[x_ch].values, data[y_ch].values
    mask = (parent_mask.values if parent_mask is not None else np.ones(len(data), dtype=bool)) & \
           np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 100:
        return 0.0
    xt, yt = biex(x[mask]), biex(y[mask])
    gate_mask = apply_gate(data, x_ch, y_ch, polygon, parent_mask)
    in_gate = gate_mask.values[mask]
    if in_gate.sum() < 10 or (~in_gate).sum() < 10:
        return 50.0
    mean_in = np.array([np.mean(xt[in_gate]), np.mean(yt[in_gate])])
    mean_out = np.array([np.mean(xt[~in_gate]), np.mean(yt[~in_gate])])
    std_in = np.array([np.std(xt[in_gate]), np.std(yt[in_gate])])
    separation = np.linalg.norm(mean_in - mean_out) / (np.mean(std_in) + 1e-10)
    sep_score = min(100, separation * 25)
    var_in = np.var(xt[in_gate]) + np.var(yt[in_gate])
    var_total = np.var(xt) + np.var(yt)
    compact_score = max(0, min(100, (1 - var_in / (var_total + 1e-10)) * 100))
    prop = in_gate.sum() / len(in_gate)
    prop_score = 100 if 0.1 <= prop <= 0.9 else (prop * 1000 if prop < 0.1 else (1 - prop) * 1000)
    return float(np.clip(0.4 * sep_score + 0.4 * compact_score + 0.2 * prop_score, 0, 100))


def remove_outliers(X, contamination=0.03):
    if len(X) < 100:
        return X, np.ones(len(X), dtype=bool)
    try:
        detector = EllipticEnvelope(contamination=contamination, random_state=42)
        mask = detector.fit_predict(X) == 1
        return X[mask], mask
    except:
        return X, np.ones(len(X), dtype=bool)


def auto_gate_hexagon_robust(data, x_ch, y_ch, parent_mask=None, mode='main'):
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
        if len(X) > 500:
            X_clean, _ = remove_outliers(X)
            if len(X_clean) < 100:
                X_clean = X
        else:
            X_clean = X

        # BIC pour sélection du nombre de composantes
        best_bic, best_n = np.inf, 2
        for n in range(2, 4):
            try:
                gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42, n_init=3)
                gmm.fit(X_clean)
                bic = gmm.bic(X_clean)
                if bic < best_bic:
                    best_bic, best_n = bic, n
            except:
                pass

        # Fit GMM
        best_gmm, best_score = None, -np.inf
        for init in range(5):
            try:
                gmm = GaussianMixture(n_components=best_n, covariance_type='full',
                                      random_state=42 + init, n_init=1, max_iter=200)
                gmm.fit(X_clean)
                score = gmm.score(X_clean)
                if score > best_score:
                    best_score, best_gmm = score, gmm
            except:
                pass

        if best_gmm is None:
            return None, 0.0

        labels = best_gmm.predict(X_clean)
        cluster_stats = []
        for i in range(best_n):
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
        confidence = compute_gate_confidence(data, polygon, x_ch, y_ch, parent_mask)
        return polygon, confidence
    except Exception as e:
        return None, 0.0


# ==================== VISUALISATION ====================

def get_confidence_class(confidence):
    if confidence >= 70:
        return "confidence-high", "🟢"
    elif confidence >= 40:
        return "confidence-medium", "🟡"
    else:
        return "confidence-low", "🔴"


def create_plot(data, x_ch, y_ch, x_label, y_label, title, polygon, parent_mask,
                gate_name, confidence=None, compact=False):
    """Crée le graphique Plotly"""
    if x_ch is None or y_ch is None:
        fig = go.Figure()
        fig.add_annotation(text="Canal non trouvé", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=300 if compact else 400)
        return fig, 0, 0.0

    x, y = data[x_ch].values, data[y_ch].values
    mask = (parent_mask.values if parent_mask is not None else np.ones(len(data), dtype=bool)) & \
           np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    n_parent = mask.sum()

    if n_parent == 0:
        fig = go.Figure()
        fig.add_annotation(text="Pas de données", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=300 if compact else 400)
        return fig, 0, 0.0

    xt, yt = biex(x[mask]), biex(y[mask])
    n_display = min(8000 if compact else 12000, len(xt))
    if len(xt) > n_display:
        idx = np.random.choice(len(xt), n_display, replace=False)
        xd, yd = xt[idx], yt[idx]
    else:
        xd, yd = xt, yt

    fig = go.Figure()

    # Points
    try:
        from scipy.stats import gaussian_kde
        if len(xd) > 500:
            xy = np.vstack([xd, yd])
            kde_idx = np.random.choice(len(xd), min(2000, len(xd)), replace=False)
            kde = gaussian_kde(xy[:, kde_idx])
            colors = kde(xy)
        else:
            colors = yd
    except:
        colors = yd

    fig.add_trace(go.Scattergl(
        x=xd, y=yd, mode='markers',
        marker=dict(size=2 if compact else 3, color=colors, colorscale='Viridis', opacity=0.5),
        hoverinfo='skip', name='Events'
    ))

    n_in, pct = 0, 0.0
    if polygon and len(polygon) >= 3:
        gate_mask = apply_gate(data, x_ch, y_ch, polygon, parent_mask)
        n_in = int(gate_mask.sum())
        pct = float(n_in / n_parent * 100) if n_parent > 0 else 0.0

        px = [p[0] for p in polygon] + [polygon[0][0]]
        py = [p[1] for p in polygon] + [polygon[0][1]]

        if confidence is not None and confidence >= 70:
            fill_color, line_color = 'rgba(40, 167, 69, 0.15)', '#28a745'
        elif confidence is not None and confidence >= 40:
            fill_color, line_color = 'rgba(255, 193, 7, 0.15)', '#ffc107'
        else:
            fill_color, line_color = 'rgba(220, 53, 69, 0.15)', '#dc3545'

        fig.add_trace(go.Scatter(x=px, y=py, fill='toself', fillcolor=fill_color,
                                  line=dict(color=line_color, width=2), mode='lines', name='Gate'))

        if not compact:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in polygon], y=[p[1] for p in polygon],
                mode='markers+text',
                marker=dict(size=12, color='white', line=dict(color=line_color, width=2)),
                text=[str(i+1) for i in range(len(polygon))],
                textposition='middle center',
                textfont=dict(size=9, color=line_color),
                name='Vertices'
            ))

        cx_p, cy_p = np.mean([p[0] for p in polygon]), np.mean([p[1] for p in polygon])
        conf_text = f"<br>{confidence:.0f}%" if confidence is not None else ""
        fig.add_annotation(
            x=cx_p, y=cy_p,
            text=f"<b>{gate_name}</b><br>{pct:.1f}%{conf_text}",
            showarrow=False, font=dict(size=9 if compact else 11),
            bgcolor='rgba(255,255,255,0.9)', bordercolor=line_color, borderwidth=1
        )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b> (n={n_parent:,})", x=0.5, font=dict(size=11 if compact else 14)),
        xaxis_title=x_label, yaxis_title=y_label,
        showlegend=False, height=280 if compact else 400,
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor='#fafafa',
        xaxis=dict(showgrid=True, gridcolor='#e0e0e0'),
        yaxis=dict(showgrid=True, gridcolor='#e0e0e0')
    )
    return fig, n_in, pct


def create_summary_figure(figures_data, filename):
    """Crée une figure récapitulative avec tous les gates"""
    n_gates = len(figures_data)
    n_cols = 3
    n_rows = (n_gates + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[d['title'] for d in figures_data],
        horizontal_spacing=0.08, vertical_spacing=0.12
    )

    for idx, gate_data in enumerate(figures_data):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        if gate_data['xd'] is not None and len(gate_data['xd']) > 0:
            fig.add_trace(go.Scattergl(
                x=gate_data['xd'], y=gate_data['yd'],
                mode='markers',
                marker=dict(size=2, color=gate_data['yd'], colorscale='Viridis', opacity=0.4),
                showlegend=False
            ), row=row, col=col)

            if gate_data['polygon']:
                px = [p[0] for p in gate_data['polygon']] + [gate_data['polygon'][0][0]]
                py = [p[1] for p in gate_data['polygon']] + [gate_data['polygon'][0][1]]

                conf = gate_data.get('confidence', 50)
                if conf >= 70:
                    line_color = '#28a745'
                elif conf >= 40:
                    line_color = '#ffc107'
                else:
                    line_color = '#dc3545'

                fig.add_trace(go.Scatter(
                    x=px, y=py, fill='toself',
                    fillcolor=f'rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.2)',
                    line=dict(color=line_color, width=2), mode='lines', showlegend=False
                ), row=row, col=col)

        fig.update_xaxes(title_text=gate_data['x_label'], row=row, col=col, title_font=dict(size=10))
        fig.update_yaxes(title_text=gate_data['y_label'], row=row, col=col, title_font=dict(size=10))

    fig.update_layout(
        title=dict(text=f"<b>Gating Summary - {filename}</b>", x=0.5, font=dict(size=16)),
        height=300 * n_rows,
        showlegend=False,
        plot_bgcolor='white'
    )
    return fig


def fig_to_bytes(fig, format='png', width=1200, height=800):
    """Convertit une figure Plotly en bytes pour téléchargement"""
    try:
        import kaleido
        return fig.to_image(format=format, width=width, height=height, scale=2)
    except ImportError:
        return None


# ==================== POINT EDITOR ====================

def render_point_editor(gkey, poly, gate_name):
    st.markdown(f"**Éditer {gate_name}**")
    modified = False
    new_poly = list(poly)

    # Afficher en 2 colonnes pour 6 points
    for row in range(3):
        cols = st.columns(4)
        for col_idx in range(2):
            i = row * 2 + col_idx
            if i < len(poly):
                with cols[col_idx * 2]:
                    new_x = st.number_input(f"P{i+1} X", value=float(poly[i][0]), step=2.0,
                                           key=f"x_{gkey}_{i}", label_visibility="collapsed")
                with cols[col_idx * 2 + 1]:
                    new_y = st.number_input(f"P{i+1} Y", value=float(poly[i][1]), step=2.0,
                                           key=f"y_{gkey}_{i}", label_visibility="collapsed")
                if new_x != poly[i][0] or new_y != poly[i][1]:
                    new_poly[i] = (new_x, new_y)
                    modified = True

    # Boutons rapides
    c1, c2, c3, c4 = st.columns(4)
    step = 10
    if c1.button("⬆️", key=f"up_{gkey}"):
        return move_polygon(poly, 0, step)
    if c2.button("⬇️", key=f"dn_{gkey}"):
        return move_polygon(poly, 0, -step)
    if c3.button("⬅️", key=f"lt_{gkey}"):
        return move_polygon(poly, -step, 0)
    if c4.button("➡️", key=f"rt_{gkey}"):
        return move_polygon(poly, step, 0)

    c5, c6, c7, c8 = st.columns(4)
    if c5.button("➕", key=f"grow_{gkey}"):
        return scale_polygon(poly, 1.1)
    if c6.button("➖", key=f"shrink_{gkey}"):
        return scale_polygon(poly, 0.9)
    if c7.button("↻", key=f"rotcw_{gkey}"):
        return rotate_polygon(poly, 15)
    if c8.button("↺", key=f"rotccw_{gkey}"):
        return rotate_polygon(poly, -15)

    return new_poly if modified else None


# ==================== MAIN APPLICATION ====================

st.markdown('<h1 class="main-header">🔬 FACS - Complete Gating v6</h1>', unsafe_allow_html=True)

learned = load_learned_params()
n_learned = learned.get('n_corrections', 0)
if n_learned > 0:
    st.success(f"🧠 {n_learned} correction(s) apprises")

# Gestion apprentissage
with st.expander("🧠 Gérer l'apprentissage", expanded=False):
    if n_learned == 0:
        st.info("Aucune correction enregistrée")
    else:
        params = load_learned_params()
        for gn, gd in params.get('gates', {}).items():
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{gn}**: {gd.get('n_samples', 0)} corrections")
            if col2.button("🗑️", key=f"del_{gn}"):
                del params['gates'][gn]
                params['n_corrections'] = max(0, params['n_corrections'] - gd.get('n_samples', 0))
                save_learned_params(params)
                st.rerun()
        if st.button("🗑️ Tout réinitialiser"):
            save_learned_params({'version': 2, 'n_corrections': 0, 'gates': {}})
            st.rerun()

# Session state
for key in ['reader', 'data', 'channels', 'polygons', 'original_polygons',
            'auto_done', 'confidences', 'undo_stack', 'figures_data']:
    if key not in st.session_state:
        st.session_state[key] = {} if key in ['channels', 'polygons', 'original_polygons', 'confidences'] else \
                                 ([] if key in ['undo_stack', 'figures_data'] else (False if key == 'auto_done' else None))

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
            st.session_state.confidences = {}
            st.session_state.auto_done = False
            st.session_state.figures_data = []

            cols = list(reader.data.columns)
            st.session_state.channels = {
                'FSC-A': find_channel(cols, ['FSC-A']),
                'FSC-H': find_channel(cols, ['FSC-H']),
                'SSC-A': find_channel(cols, ['SSC-A']),
                'LiveDead': find_channel(cols, ['LiveDead', 'Viab', 'Aqua', 'Live', 'L/D', 'LD']),
                'hCD45': find_channel(cols, ['PerCP', 'CD45', 'hCD45']),
                'CD3': find_channel(cols, ['AF488', 'FITC', 'CD3']),
                'CD19': find_channel(cols, ['PE-Fire', 'CD19', 'PE-CF594']),
                'CD4': find_channel(cols, ['BV650', 'CD4', 'APC-Cy7']),
                'CD8': find_channel(cols, ['BUV805', 'CD8', 'APC']),
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

    with st.expander("📋 Canaux détectés"):
        for name, canal in ch.items():
            st.write(f"{'✅' if canal else '❌'} **{name}**: {canal or 'Non trouvé'}")

    if ch['FSC-A'] is None or ch['SSC-A'] is None:
        st.error("❌ FSC-A ou SSC-A non trouvé!")
        st.stop()

    st.markdown("---")

    # AUTO-GATING
    if not st.session_state.auto_done:
        if st.button("🚀 LANCER L'AUTO-GATING COMPLET", type="primary", use_container_width=True):
            progress = st.progress(0, "Initialisation...")
            polygons = st.session_state.polygons
            confidences = st.session_state.confidences
            original = st.session_state.original_polygons

            # 1. Cells
            progress.progress(5, "Gate Cells...")
            poly, conf = auto_gate_hexagon_robust(data, ch['FSC-A'], ch['SSC-A'], None, 'main')
            poly = apply_learned_adj(poly, 'cells')
            polygons['cells'], confidences['cells'] = poly, conf
            original['cells'] = list(poly) if poly else None

            # 2. Singlets
            progress.progress(15, "Gate Singlets...")
            if ch['FSC-H'] and poly:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], poly, None)
                poly2, conf2 = auto_gate_hexagon_robust(data, ch['FSC-A'], ch['FSC-H'], cells_m, 'main')
                poly2 = apply_learned_adj(poly2, 'singlets')
            else:
                poly2, conf2 = None, 0.0
            polygons['singlets'], confidences['singlets'] = poly2, conf2
            original['singlets'] = list(poly2) if poly2 else None

            # 3. Live
            progress.progress(25, "Gate Live...")
            if ch['LiveDead']:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], polygons['cells'], None)
                sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], polygons['singlets'], cells_m) if polygons['singlets'] else cells_m
                poly3, conf3 = auto_gate_hexagon_robust(data, ch['LiveDead'], ch['SSC-A'], sing_m, 'low_x')
                poly3 = apply_learned_adj(poly3, 'live')
            else:
                poly3, conf3 = None, 0.0
            polygons['live'], confidences['live'] = poly3, conf3
            original['live'] = list(poly3) if poly3 else None

            # 4. hCD45+
            progress.progress(35, "Gate hCD45+...")
            if ch['hCD45']:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], polygons['cells'], None)
                sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], polygons['singlets'], cells_m) if polygons['singlets'] else cells_m
                live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'], polygons['live'], sing_m) if polygons['live'] else sing_m
                poly4, conf4 = auto_gate_hexagon_robust(data, ch['hCD45'], ch['SSC-A'], live_m, 'high_x')
                poly4 = apply_learned_adj(poly4, 'hcd45')
            else:
                poly4, conf4 = None, 0.0
            polygons['hcd45'], confidences['hcd45'] = poly4, conf4
            original['hcd45'] = list(poly4) if poly4 else None

            # Parent mask pour marqueurs
            cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], polygons['cells'], None)
            sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], polygons['singlets'], cells_m) if polygons['singlets'] else cells_m
            live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'], polygons['live'], sing_m) if polygons['live'] else sing_m
            hcd45_m = apply_gate(data, ch['hCD45'], ch['SSC-A'], polygons['hcd45'], live_m) if polygons['hcd45'] else live_m

            # 5. CD3+
            progress.progress(50, "Gate CD3+...")
            if ch['CD3']:
                poly5, conf5 = auto_gate_hexagon_robust(data, ch['CD3'], ch['SSC-A'], hcd45_m, 'high_x')
                poly5 = apply_learned_adj(poly5, 'cd3')
            else:
                poly5, conf5 = None, 0.0
            polygons['cd3'], confidences['cd3'] = poly5, conf5
            original['cd3'] = list(poly5) if poly5 else None

            # 6. CD19+
            progress.progress(60, "Gate CD19+...")
            if ch['CD19']:
                poly6, conf6 = auto_gate_hexagon_robust(data, ch['CD19'], ch['SSC-A'], hcd45_m, 'high_x')
                poly6 = apply_learned_adj(poly6, 'cd19')
            else:
                poly6, conf6 = None, 0.0
            polygons['cd19'], confidences['cd19'] = poly6, conf6
            original['cd19'] = list(poly6) if poly6 else None

            # CD3+ mask pour CD4/CD8
            cd3_m = apply_gate(data, ch['CD3'], ch['SSC-A'], polygons['cd3'], hcd45_m) if polygons['cd3'] else hcd45_m

            # 7. CD4+
            progress.progress(75, "Gate CD4+...")
            if ch['CD4']:
                poly7, conf7 = auto_gate_hexagon_robust(data, ch['CD4'], ch['SSC-A'], cd3_m, 'high_x')
                poly7 = apply_learned_adj(poly7, 'cd4')
            else:
                poly7, conf7 = None, 0.0
            polygons['cd4'], confidences['cd4'] = poly7, conf7
            original['cd4'] = list(poly7) if poly7 else None

            # 8. CD8+
            progress.progress(90, "Gate CD8+...")
            if ch['CD8']:
                poly8, conf8 = auto_gate_hexagon_robust(data, ch['CD8'], ch['SSC-A'], cd3_m, 'high_x')
                poly8 = apply_learned_adj(poly8, 'cd8')
            else:
                poly8, conf8 = None, 0.0
            polygons['cd8'], confidences['cd8'] = poly8, conf8
            original['cd8'] = list(poly8) if poly8 else None

            progress.progress(100, "Terminé!")
            st.session_state.auto_done = True
            st.rerun()

    # AFFICHAGE DES GATES
    if st.session_state.auto_done:
        polygons = st.session_state.polygons
        confidences = st.session_state.confidences

        # Recalcul des masques
        cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], polygons.get('cells'), None)
        sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], polygons.get('singlets'), cells_m) if polygons.get('singlets') else cells_m
        live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'], polygons.get('live'), sing_m) if polygons.get('live') else sing_m
        hcd45_m = apply_gate(data, ch['hCD45'], ch['SSC-A'], polygons.get('hcd45'), live_m) if polygons.get('hcd45') else live_m
        cd3_m = apply_gate(data, ch['CD3'], ch['SSC-A'], polygons.get('cd3'), hcd45_m) if polygons.get('cd3') else hcd45_m

        # Configuration de tous les gates
        all_gates = [
            ('cells', 'Cells', ch['FSC-A'], ch['SSC-A'], 'FSC-A', 'SSC-A', 'Ungated → Cells', None),
            ('singlets', 'Singlets', ch['FSC-A'], ch['FSC-H'], 'FSC-A', 'FSC-H', 'Cells → Singlets', cells_m),
            ('live', 'Live', ch['LiveDead'], ch['SSC-A'], 'Live/Dead', 'SSC-A', 'Singlets → Live', sing_m),
            ('hcd45', 'hCD45+', ch['hCD45'], ch['SSC-A'], 'hCD45', 'SSC-A', 'Live → hCD45+', live_m),
            ('cd3', 'CD3+', ch['CD3'], ch['SSC-A'], 'CD3', 'SSC-A', 'hCD45+ → CD3+', hcd45_m),
            ('cd19', 'CD19+', ch['CD19'], ch['SSC-A'], 'CD19', 'SSC-A', 'hCD45+ → CD19+', hcd45_m),
            ('cd4', 'CD4+', ch['CD4'], ch['SSC-A'], 'CD4', 'SSC-A', 'CD3+ → CD4+', cd3_m),
            ('cd8', 'CD8+', ch['CD8'], ch['SSC-A'], 'CD8', 'SSC-A', 'CD3+ → CD8+', cd3_m),
        ]

        # Filtrer les gates avec canaux disponibles
        valid_gates = [(g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]) for g in all_gates if g[2] is not None and g[3] is not None]

        # Tabs pour organisation
        tab1, tab2, tab3 = st.tabs(["📊 Gates Individuels", "🗺️ Vue Récapitulative", "📥 Export"])

        with tab1:
            stats = []
            figures_data = []

            # Affichage en grille 2x4 ou 3x3
            n_gates = len(valid_gates)
            n_cols = 3

            for row_start in range(0, n_gates, n_cols):
                cols_display = st.columns(n_cols)
                for col_idx in range(n_cols):
                    gate_idx = row_start + col_idx
                    if gate_idx >= n_gates:
                        break

                    gkey, gname, x_ch, y_ch, x_label, y_label, title, parent_mask = valid_gates[gate_idx]

                    with cols_display[col_idx]:
                        conf = confidences.get(gkey, 0)
                        _, conf_icon = get_confidence_class(conf)
                        st.markdown(f"**{gname}** {conf_icon}")

                        fig, n_in, pct = create_plot(
                            data, x_ch, y_ch, x_label, y_label, title,
                            polygons.get(gkey), parent_mask, gname, conf, compact=True
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{gkey}")

                        # Données pour export
                        x, y = data[x_ch].values, data[y_ch].values
                        mask = (parent_mask.values if parent_mask is not None else np.ones(len(data), dtype=bool)) & \
                               np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                        n_parent = mask.sum()
                        if n_parent > 0:
                            xt, yt = biex(x[mask]), biex(y[mask])
                            n_sample = min(5000, len(xt))
                            idx = np.random.choice(len(xt), n_sample, replace=False) if len(xt) > n_sample else np.arange(len(xt))
                            figures_data.append({
                                'title': f"{gname} ({pct:.1f}%)",
                                'xd': xt[idx], 'yd': yt[idx],
                                'x_label': x_label, 'y_label': y_label,
                                'polygon': polygons.get(gkey),
                                'confidence': conf
                            })
                        else:
                            figures_data.append({
                                'title': gname, 'xd': None, 'yd': None,
                                'x_label': x_label, 'y_label': y_label,
                                'polygon': None, 'confidence': 0
                            })

                        parent_name = title.split('→')[0].strip()
                        stats.append((gname, parent_name, n_in, pct, conf, n_parent))

                        # Éditeur
                        poly = polygons.get(gkey)
                        if poly:
                            with st.expander(f"✏️ Modifier", expanded=False):
                                new_poly = render_point_editor(gkey, poly, gname)
                                if new_poly is not None:
                                    st.session_state.polygons[gkey] = new_poly
                                    st.session_state.confidences[gkey] = compute_gate_confidence(data, new_poly, x_ch, y_ch, parent_mask)
                                    st.rerun()

            st.session_state.figures_data = figures_data

            st.markdown("---")

            # Actions
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("💾 Sauvegarder apprentissage", type="primary", use_container_width=True):
                    n_saved = 0
                    for gname in polygons:
                        curr = polygons.get(gname)
                        orig = st.session_state.original_polygons.get(gname)
                        conf = confidences.get(gname)
                        if curr and orig and str(curr) != str(orig):
                            update_learned_params(gname, orig, curr, conf)
                            n_saved += 1
                    st.success(f"✅ {n_saved} correction(s)") if n_saved else st.info("Aucune modif")

            with col_b:
                if st.button("🔃 Reset tous les gates", use_container_width=True):
                    st.session_state.polygons = {k: list(v) if v else None for k, v in st.session_state.original_polygons.items()}
                    st.rerun()

            with col_c:
                if st.button("🔄 Rafraîchir", use_container_width=True):
                    st.rerun()

            # Tableau résumé
            st.markdown("### 📊 Statistiques")
            df = pd.DataFrame(stats, columns=['Population', 'Parent', 'Count', '% Parent', 'Confiance', 'N_Parent'])
            df['% Total'] = (df['Count'] / n_total * 100).round(2)
            df['% Parent'] = df['% Parent'].round(1)
            df['Confiance'] = df['Confiance'].round(0).astype(int).astype(str) + '%'
            st.dataframe(df[['Population', 'Parent', 'Count', '% Parent', '% Total', 'Confiance']],
                        use_container_width=True, hide_index=True)

        with tab2:
            st.markdown("### 🗺️ Vue Récapitulative de Tous les Gates")

            if st.session_state.figures_data:
                summary_fig = create_summary_figure(st.session_state.figures_data, reader.filename)
                st.plotly_chart(summary_fig, use_container_width=True)

            # Hiérarchie des gates
            st.markdown("### 🌳 Hiérarchie du Gating")
            hierarchy = """
            ```
            📁 All Events
            └── 🔵 Cells (FSC-A vs SSC-A)
                └── 🔵 Singlets (FSC-A vs FSC-H)
                    └── 🟢 Live (LiveDead vs SSC-A)
                        └── 🟣 hCD45+ (hCD45 vs SSC-A)
                            ├── 🔴 CD3+ T cells (CD3 vs SSC-A)
                            │   ├── 🟠 CD4+ Helper T (CD4 vs SSC-A)
                            │   └── 🟡 CD8+ Cytotoxic T (CD8 vs SSC-A)
                            └── 🔵 CD19+ B cells (CD19 vs SSC-A)
            ```
            """
            st.markdown(hierarchy)

        with tab3:
            st.markdown("### 📥 Export des Données et Graphiques")

            # Export données
            st.markdown("#### 📊 Données")
            col1, col2 = st.columns(2)

            export_df = df.drop('Confiance', axis=1) if 'df' in dir() else pd.DataFrame()
            col1.download_button(
                "📥 Statistiques CSV",
                export_df.to_csv(index=False) if len(export_df) > 0 else "",
                f"{reader.filename}_stats.csv",
                "text/csv",
                use_container_width=True
            )

            buf = io.BytesIO()
            if len(export_df) > 0:
                export_df.to_excel(buf, index=False, engine='openpyxl')
            buf.seek(0)
            col2.download_button(
                "📥 Statistiques Excel",
                buf.getvalue(),
                f"{reader.filename}_stats.xlsx",
                use_container_width=True
            )

            # Export graphiques
            st.markdown("#### 🖼️ Graphiques")
            st.info("💡 Pour exporter les graphiques, utilisez le menu Plotly (icône appareil photo) sur chaque graphique, ou exportez la vue récapitulative ci-dessous.")

            # Export figure récapitulative
            if st.session_state.figures_data:
                summary_fig = create_summary_figure(st.session_state.figures_data, reader.filename)

                col1, col2, col3 = st.columns(3)

                # HTML interactif
                html_buffer = io.StringIO()
                summary_fig.write_html(html_buffer, include_plotlyjs='cdn')
                html_bytes = html_buffer.getvalue().encode()
                col1.download_button(
                    "📥 HTML Interactif",
                    html_bytes,
                    f"{reader.filename}_summary.html",
                    "text/html",
                    use_container_width=True
                )

                # JSON (pour réimport)
                json_str = summary_fig.to_json()
                col2.download_button(
                    "📥 JSON (Plotly)",
                    json_str,
                    f"{reader.filename}_summary.json",
                    "application/json",
                    use_container_width=True
                )

                # PNG (si kaleido disponible)
                try:
                    png_bytes = summary_fig.to_image(format='png', width=1600, height=1200, scale=2)
                    col3.download_button(
                        "📥 PNG (Image)",
                        png_bytes,
                        f"{reader.filename}_summary.png",
                        "image/png",
                        use_container_width=True
                    )
                except Exception as e:
                    col3.warning("PNG: installez kaleido")

                # Export individuel
                st.markdown("#### 📁 Export Individuel des Gates")

                gate_to_export = st.selectbox(
                    "Sélectionner un gate",
                    [g[1] for g in valid_gates],
                    key="gate_export_select"
                )

                # Trouver le gate
                for gkey, gname, x_ch, y_ch, x_label, y_label, title, parent_mask in valid_gates:
                    if gname == gate_to_export:
                        fig_single, _, _ = create_plot(
                            data, x_ch, y_ch, x_label, y_label, title,
                            polygons.get(gkey), parent_mask, gname,
                            confidences.get(gkey, 0), compact=False
                        )

                        col1, col2 = st.columns(2)

                        html_buf = io.StringIO()
                        fig_single.write_html(html_buf, include_plotlyjs='cdn')
                        col1.download_button(
                            f"📥 {gname} HTML",
                            html_buf.getvalue().encode(),
                            f"{reader.filename}_{gkey}.html",
                            "text/html",
                            use_container_width=True
                        )

                        try:
                            png_single = fig_single.to_image(format='png', width=1000, height=800, scale=2)
                            col2.download_button(
                                f"📥 {gname} PNG",
                                png_single,
                                f"{reader.filename}_{gkey}.png",
                                "image/png",
                                use_container_width=True
                            )
                        except:
                            col2.info("PNG: kaleido requis")

                        break

else:
    st.markdown("""
    <div class="info-box">
    <h3>🔬 FACS Complete Gating v6</h3>
    <ul>
    <li>🎯 <b>8 Gates complets</b>: Cells, Singlets, Live, hCD45, CD3, CD19, CD4, CD8</li>
    <li>📊 <b>Vue récapitulative</b>: Tous les gates en une seule figure</li>
    <li>📥 <b>Export multi-format</b>: PNG, HTML, Excel, CSV</li>
    <li>🧠 <b>Apprentissage</b>: Les corrections améliorent les futurs gatings</li>
    </ul>
    <p>Uploadez un fichier FCS pour commencer</p>
    </div>
    """, unsafe_allow_html=True)

st.caption(f"v6 | 🧠 {n_learned} corrections | Complete Gating + Export")
