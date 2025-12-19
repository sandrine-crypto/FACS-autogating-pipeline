#!/usr/bin/env python3
"""
FACS Autogating - Gates Hexagonaux v5
Optimisations:
- Edition interactive améliorée (tous les points visibles, drag intuitif)
- GMM plus fiable (BIC, initialisations multiples, détection outliers)
- Métriques de confiance pour les gates
- Apprentissage amélioré avec scaling et rotation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import RobustScaler
from scipy import stats as scipy_stats
from pathlib import Path
import tempfile
import io
import json
import os
import flowio

st.set_page_config(page_title="FACS - Hexagones v5", page_icon="🔬", layout="wide")

LEARNED_PARAMS_FILE = "learned_gating_params.json"

st.markdown("""
<style>
.main-header { font-size: 1.8rem; color: #2c3e50; text-align: center; margin-bottom: 0.5rem; }
.info-box { background: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid #0066cc; margin: 0.5rem 0; }
.confidence-high { color: #28a745; font-weight: bold; }
.confidence-medium { color: #ffc107; font-weight: bold; }
.confidence-low { color: #dc3545; font-weight: bold; }
.point-editor { background: #f8f9fa; padding: 0.5rem; border-radius: 0.3rem; margin: 0.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ==================== APPRENTISSAGE AMELIORE ====================

def load_learned_params():
    """Charge les paramètres appris avec structure enrichie"""
    if os.path.exists(LEARNED_PARAMS_FILE):
        try:
            with open(LEARNED_PARAMS_FILE, 'r') as f:
                params = json.load(f)
                # Migration vers nouvelle structure si nécessaire
                if 'version' not in params:
                    params['version'] = 2
                    for gate in params.get('gates', {}).values():
                        if 'scale_factor' not in gate:
                            gate['scale_factor'] = 1.0
                        if 'rotation' not in gate:
                            gate['rotation'] = 0.0
                        if 'confidence_history' not in gate:
                            gate['confidence_history'] = []
                return params
        except:
            pass
    return {'version': 2, 'n_corrections': 0, 'gates': {}}


def save_learned_params(params):
    """Sauvegarde les paramètres avec backup"""
    # Backup
    if os.path.exists(LEARNED_PARAMS_FILE):
        try:
            os.rename(LEARNED_PARAMS_FILE, LEARNED_PARAMS_FILE + '.bak')
        except:
            pass
    with open(LEARNED_PARAMS_FILE, 'w') as f:
        json.dump(params, f, indent=2)


def update_learned_params(gate_name, original_polygon, corrected_polygon, confidence=None):
    """
    Apprentissage amélioré: translation, scaling, et rotation
    """
    params = load_learned_params()

    if gate_name not in params['gates']:
        params['gates'][gate_name] = {
            'avg_adjustment': {'x': 0, 'y': 0},
            'scale_factor': 1.0,
            'rotation': 0.0,
            'n_samples': 0,
            'confidence_history': []
        }

    gate_params = params['gates'][gate_name]

    # Calculer les transformations
    orig_arr = np.array(original_polygon)
    corr_arr = np.array(corrected_polygon)

    orig_center = np.mean(orig_arr, axis=0)
    corr_center = np.mean(corr_arr, axis=0)

    # Translation
    dx = float(corr_center[0] - orig_center[0])
    dy = float(corr_center[1] - orig_center[1])

    # Scaling (ratio des distances moyennes au centre)
    orig_dists = np.linalg.norm(orig_arr - orig_center, axis=1)
    corr_dists = np.linalg.norm(corr_arr - corr_center, axis=1)
    scale = float(np.mean(corr_dists) / (np.mean(orig_dists) + 1e-10))

    # Mise à jour avec moyenne mobile exponentielle
    gate_params['n_samples'] += 1
    n = gate_params['n_samples']
    alpha = 2 / (n + 1)  # EMA decay

    gate_params['avg_adjustment']['x'] = (1 - alpha) * gate_params['avg_adjustment']['x'] + alpha * dx
    gate_params['avg_adjustment']['y'] = (1 - alpha) * gate_params['avg_adjustment']['y'] + alpha * dy
    gate_params['scale_factor'] = (1 - alpha) * gate_params['scale_factor'] + alpha * scale

    # Historique de confiance
    if confidence is not None:
        gate_params['confidence_history'].append(float(confidence))
        # Garder les 20 dernières valeurs
        gate_params['confidence_history'] = gate_params['confidence_history'][-20:]

    params['n_corrections'] += 1
    save_learned_params(params)


def apply_learned_adj(polygon, gate_name):
    """Applique les ajustements appris (translation + scaling)"""
    if polygon is None:
        return None

    params = load_learned_params()
    if gate_name not in params['gates']:
        return polygon

    gate = params['gates'][gate_name]
    adj = gate.get('avg_adjustment', {'x': 0, 'y': 0})
    scale = gate.get('scale_factor', 1.0)

    # Seuil minimum pour appliquer
    if abs(adj['x']) < 0.5 and abs(adj['y']) < 0.5 and abs(scale - 1.0) < 0.02:
        return polygon

    # Appliquer scaling puis translation
    poly_arr = np.array(polygon)
    center = np.mean(poly_arr, axis=0)

    # Scale autour du centre
    scaled = center + (poly_arr - center) * scale

    # Translate
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
    """Recherche de canal avec priorité aux correspondances exactes"""
    # D'abord chercher correspondance exacte
    for col in columns:
        for kw in keywords:
            if col.upper() == kw.upper():
                return col
    # Ensuite correspondance partielle
    for col in columns:
        col_upper = col.upper()
        for kw in keywords:
            if kw.upper() in col_upper:
                return col
    return None


# ==================== TRANSFORMATIONS ====================

def biex(x, width=150, scale=50):
    """Transformation biexponentielle paramétrable"""
    return np.arcsinh(np.asarray(x, float) / width) * scale


def inverse_biex(y, width=150, scale=50):
    """Inverse de la transformation biexponentielle"""
    return np.sinh(np.asarray(y, float) / scale) * width


# ==================== GEOMETRIE ====================

def create_hexagon(center_x, center_y, radius_x, radius_y):
    """Crée un hexagone régulier"""
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    return [(float(center_x + radius_x * np.cos(a)), float(center_y + radius_y * np.sin(a))) for a in angles]


def point_in_polygon(x, y, polygon):
    """Ray casting algorithm optimisé"""
    if polygon is None or len(polygon) < 3:
        return np.zeros(len(x), dtype=bool)

    n = len(polygon)
    px = np.array([p[0] for p in polygon], dtype=np.float64)
    py = np.array([p[1] for p in polygon], dtype=np.float64)

    inside = np.zeros(len(x), dtype=bool)
    j = n - 1

    for i in range(n):
        # Éviter division par zéro
        dy = py[j] - py[i]
        if abs(dy) < 1e-10:
            dy = 1e-10

        cond = ((py[i] > y) != (py[j] > y)) & \
               (x < (px[j] - px[i]) * (y - py[i]) / dy + px[i])
        inside ^= cond
        j = i

    return inside


def apply_gate(data, x_ch, y_ch, polygon, parent_mask=None):
    """Applique un gate polygonal avec gestion robuste"""
    if x_ch is None or y_ch is None or polygon is None or len(polygon) < 3:
        return pd.Series(False, index=data.index)

    x = data[x_ch].values
    y = data[y_ch].values

    if parent_mask is not None:
        base = parent_mask.values.copy()
    else:
        base = np.ones(len(data), dtype=bool)

    # Valeurs valides
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & base

    if not valid.any():
        return pd.Series(False, index=data.index)

    xt = biex(x)
    yt = biex(y)

    in_poly = point_in_polygon(xt, yt, polygon)
    result = valid & in_poly

    return pd.Series(result, index=data.index)


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
    """Rotation du polygone autour de son centre"""
    if polygon is None:
        return None

    angle_rad = np.radians(angle_deg)
    center = np.mean(polygon, axis=0)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    rotated = []
    for p in polygon:
        dx, dy = p[0] - center[0], p[1] - center[1]
        new_x = center[0] + dx * cos_a - dy * sin_a
        new_y = center[1] + dx * sin_a + dy * cos_a
        rotated.append((float(new_x), float(new_y)))

    return rotated


# ==================== AUTO-GATING AMELIORE ====================

def compute_gate_confidence(data, polygon, x_ch, y_ch, parent_mask=None):
    """
    Calcule un score de confiance pour le gate (0-100)
    Basé sur: séparation des clusters, densité, forme
    """
    if polygon is None or x_ch is None or y_ch is None:
        return 0.0

    x = data[x_ch].values
    y = data[y_ch].values

    if parent_mask is not None:
        mask = parent_mask.values & np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    else:
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

    if mask.sum() < 100:
        return 0.0

    xt, yt = biex(x[mask]), biex(y[mask])

    gate_mask = apply_gate(data, x_ch, y_ch, polygon, parent_mask)
    in_gate = gate_mask.values[mask]

    if in_gate.sum() < 10 or (~in_gate).sum() < 10:
        return 50.0  # Pas assez de données pour évaluer

    # Score 1: Séparation (distance entre moyennes / écart-type)
    mean_in = np.array([np.mean(xt[in_gate]), np.mean(yt[in_gate])])
    mean_out = np.array([np.mean(xt[~in_gate]), np.mean(yt[~in_gate])])
    std_in = np.array([np.std(xt[in_gate]), np.std(yt[in_gate])])

    separation = np.linalg.norm(mean_in - mean_out) / (np.mean(std_in) + 1e-10)
    sep_score = min(100, separation * 25)

    # Score 2: Compacité (variance intra-cluster)
    var_in = np.var(xt[in_gate]) + np.var(yt[in_gate])
    var_total = np.var(xt) + np.var(yt)
    compactness = 1 - (var_in / (var_total + 1e-10))
    compact_score = max(0, min(100, compactness * 100))

    # Score 3: Proportion raisonnable (pas trop petit ni trop grand)
    prop = in_gate.sum() / len(in_gate)
    if 0.1 <= prop <= 0.9:
        prop_score = 100
    elif prop < 0.1:
        prop_score = prop * 1000
    else:
        prop_score = (1 - prop) * 1000

    # Moyenne pondérée
    confidence = 0.4 * sep_score + 0.4 * compact_score + 0.2 * prop_score

    return float(np.clip(confidence, 0, 100))


def select_optimal_n_components(X, max_components=4):
    """
    Sélectionne le nombre optimal de composantes GMM via BIC
    """
    best_bic = np.inf
    best_n = 2

    for n in range(2, max_components + 1):
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full',
                                  random_state=42, n_init=3, max_iter=200)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        except:
            continue

    return best_n


def remove_outliers(X, contamination=0.05):
    """
    Supprime les outliers avant GMM pour plus de robustesse
    """
    if len(X) < 100:
        return X, np.ones(len(X), dtype=bool)

    try:
        detector = EllipticEnvelope(contamination=contamination, random_state=42)
        mask = detector.fit_predict(X) == 1
        return X[mask], mask
    except:
        return X, np.ones(len(X), dtype=bool)


def auto_gate_hexagon_robust(data, x_ch, y_ch, parent_mask=None, mode='main',
                             remove_outliers_flag=True, adaptive_components=True):
    """
    Auto-gating amélioré avec:
    - Détection et suppression des outliers
    - Sélection adaptative du nombre de composantes
    - Initialisations multiples
    - Retourne aussi un score de confiance
    """
    if x_ch is None or y_ch is None:
        return None, 0.0

    x = data[x_ch].values
    y = data[y_ch].values

    if parent_mask is not None:
        mask = parent_mask.values & np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    else:
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

    if mask.sum() < 100:
        return None, 0.0

    xt = biex(x[mask])
    yt = biex(y[mask])
    X = np.column_stack([xt, yt])

    try:
        # Étape 1: Suppression des outliers
        if remove_outliers_flag and len(X) > 500:
            X_clean, inlier_mask = remove_outliers(X, contamination=0.03)
        else:
            X_clean, inlier_mask = X, np.ones(len(X), dtype=bool)

        if len(X_clean) < 100:
            X_clean = X
            inlier_mask = np.ones(len(X), dtype=bool)

        # Étape 2: Sélection du nombre de composantes
        if adaptive_components:
            n_components = select_optimal_n_components(X_clean, max_components=3)
        else:
            n_components = 2

        # Étape 3: Fit GMM avec initialisations multiples
        best_gmm = None
        best_score = -np.inf

        for init in range(5):  # 5 initialisations
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42 + init,
                    n_init=1,
                    max_iter=200,
                    tol=1e-4
                )
                gmm.fit(X_clean)
                score = gmm.score(X_clean)

                if score > best_score:
                    best_score = score
                    best_gmm = gmm
            except:
                continue

        if best_gmm is None:
            return None, 0.0

        labels = best_gmm.predict(X_clean)

        # Étape 4: Sélection du cluster selon le mode
        cluster_stats = []
        for i in range(n_components):
            cluster_mask = labels == i
            if cluster_mask.sum() > 0:
                cluster_stats.append({
                    'idx': i,
                    'count': cluster_mask.sum(),
                    'mean_x': np.mean(X_clean[cluster_mask, 0]),
                    'mean_y': np.mean(X_clean[cluster_mask, 1]),
                    'density': cluster_mask.sum() / (np.std(X_clean[cluster_mask, 0]) * np.std(X_clean[cluster_mask, 1]) + 1e-10)
                })

        if not cluster_stats:
            return None, 0.0

        # Sélection selon le mode
        if mode == 'main':
            # Cluster le plus peuplé
            target_idx = max(cluster_stats, key=lambda x: x['count'])['idx']
        elif mode == 'low_x':
            # Cluster avec moyenne X la plus basse
            target_idx = min(cluster_stats, key=lambda x: x['mean_x'])['idx']
        elif mode == 'high_x':
            # Cluster avec moyenne X la plus haute
            target_idx = max(cluster_stats, key=lambda x: x['mean_x'])['idx']
        elif mode == 'dense':
            # Cluster le plus dense
            target_idx = max(cluster_stats, key=lambda x: x['density'])['idx']
        else:
            target_idx = 0

        cluster_mask = labels == target_idx
        cx, cy = X_clean[cluster_mask, 0], X_clean[cluster_mask, 1]

        if len(cx) < 50:
            return None, 0.0

        # Étape 5: Calcul du gate avec robustesse
        # Utiliser médiane et IQR au lieu de moyenne/std
        center_x = np.median(cx)
        center_y = np.median(cy)

        # Rayons basés sur les percentiles (plus robuste)
        radius_x = np.percentile(np.abs(cx - center_x), 90) * 1.2
        radius_y = np.percentile(np.abs(cy - center_y), 90) * 1.2

        # Minimum radius
        radius_x = max(radius_x, 8)
        radius_y = max(radius_y, 8)

        polygon = create_hexagon(center_x, center_y, radius_x, radius_y)

        # Calculer confiance
        confidence = compute_gate_confidence(data, polygon, x_ch, y_ch, parent_mask)

        return polygon, confidence

    except Exception as e:
        st.warning(f"Erreur auto-gate: {e}")
        return None, 0.0


# ==================== VISUALISATION ====================

def get_confidence_class(confidence):
    """Retourne la classe CSS selon le niveau de confiance"""
    if confidence >= 70:
        return "confidence-high", "🟢"
    elif confidence >= 40:
        return "confidence-medium", "🟡"
    else:
        return "confidence-low", "🔴"


def create_plot(data, x_ch, y_ch, x_label, y_label, title, polygon, parent_mask,
                gate_name, confidence=None, show_density=True):
    """Crée le graphique Plotly avec visualisation améliorée"""

    if x_ch is None or y_ch is None:
        fig = go.Figure()
        fig.add_annotation(text="Canal non trouvé", x=0.5, y=0.5,
                          xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=400)
        return fig, 0, 0.0

    x = data[x_ch].values
    y = data[y_ch].values

    if parent_mask is not None:
        mask = parent_mask.values & np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    else:
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

    n_parent = mask.sum()

    if n_parent == 0:
        fig = go.Figure()
        fig.add_annotation(text="Pas de données", x=0.5, y=0.5,
                          xref="paper", yref="paper", showarrow=False)
        fig.update_layout(height=400)
        return fig, 0, 0.0

    xt = biex(x[mask])
    yt = biex(y[mask])

    # Sous-échantillonnage intelligent (stratifié)
    n_display = min(12000, len(xt))
    if len(xt) > n_display:
        idx = np.random.choice(len(xt), n_display, replace=False)
        xd, yd = xt[idx], yt[idx]
    else:
        xd, yd = xt, yt

    fig = go.Figure()

    # Scatter avec colorisation par densité locale
    if show_density and len(xd) > 100:
        # Estimation de densité simple via histogramme 2D
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([xd, yd])
            # Sous-échantillonner pour KDE si trop de points
            if len(xd) > 3000:
                kde_idx = np.random.choice(len(xd), 3000, replace=False)
                kde = gaussian_kde(xy[:, kde_idx])
            else:
                kde = gaussian_kde(xy)
            density = kde(xy)
            colors = density
        except:
            colors = yd
    else:
        colors = yd

    fig.add_trace(go.Scattergl(
        x=xd,
        y=yd,
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            colorscale='Viridis',
            opacity=0.5,
            showscale=False
        ),
        hoverinfo='skip',
        name='Events'
    ))

    n_in, pct = 0, 0.0

    if polygon and len(polygon) >= 3:
        gate_mask = apply_gate(data, x_ch, y_ch, polygon, parent_mask)
        n_in = int(gate_mask.sum())
        pct = float(n_in / n_parent * 100) if n_parent > 0 else 0.0

        # Polygone avec style amélioré
        px = [p[0] for p in polygon] + [polygon[0][0]]
        py = [p[1] for p in polygon] + [polygon[0][1]]

        # Couleur selon confiance
        if confidence is not None and confidence >= 70:
            fill_color = 'rgba(40, 167, 69, 0.15)'
            line_color = '#28a745'
        elif confidence is not None and confidence >= 40:
            fill_color = 'rgba(255, 193, 7, 0.15)'
            line_color = '#ffc107'
        else:
            fill_color = 'rgba(220, 53, 69, 0.15)'
            line_color = '#dc3545'

        fig.add_trace(go.Scatter(
            x=px,
            y=py,
            fill='toself',
            fillcolor=fill_color,
            line=dict(color=line_color, width=2.5),
            mode='lines',
            name='Gate'
        ))

        # Sommets interactifs avec meilleure visibilité
        fig.add_trace(go.Scatter(
            x=[p[0] for p in polygon],
            y=[p[1] for p in polygon],
            mode='markers+text',
            marker=dict(
                size=14,
                color='white',
                line=dict(color=line_color, width=3),
                symbol='circle'
            ),
            text=[str(i+1) for i in range(len(polygon))],
            textposition='middle center',
            textfont=dict(size=10, color=line_color, family='Arial Black'),
            name='Vertices',
            hovertemplate='<b>Point %{text}</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
        ))

        # Annotation avec confiance
        cx_poly = np.mean([p[0] for p in polygon])
        cy_poly = np.mean([p[1] for p in polygon])

        conf_text = f"<br>Conf: {confidence:.0f}%" if confidence is not None else ""

        fig.add_annotation(
            x=cx_poly, y=cy_poly,
            text=f"<b>{gate_name}</b><br>{pct:.1f}%<br>({n_in:,}){conf_text}",
            showarrow=False,
            font=dict(size=11, color='#333'),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor=line_color,
            borderwidth=2,
            borderpad=4
        )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b> (n={n_parent:,})", x=0.5),
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False,
        height=420,
        margin=dict(l=60, r=30, t=50, b=50),
        plot_bgcolor='#fafafa',
        xaxis=dict(showgrid=True, gridcolor='#e0e0e0', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#e0e0e0', zeroline=False),
    )

    return fig, n_in, pct


# ==================== EDITION DES POINTS ====================

def render_point_editor(gkey, poly, gate_name):
    """Interface d'édition des points améliorée"""

    st.markdown(f"##### ✏️ Éditer {gate_name}")

    # Afficher tous les points dans un tableau éditable
    cols_header = st.columns([1, 2, 2])
    cols_header[0].markdown("**#**")
    cols_header[1].markdown("**X**")
    cols_header[2].markdown("**Y**")

    modified = False
    new_poly = list(poly)

    for i in range(len(poly)):
        cols = st.columns([1, 2, 2])
        with cols[0]:
            st.markdown(f"**{i+1}**")
        with cols[1]:
            new_x = st.number_input(
                f"X{i+1}",
                value=float(poly[i][0]),
                step=2.0,
                format="%.1f",
                key=f"x_{gkey}_{i}",
                label_visibility="collapsed"
            )
        with cols[2]:
            new_y = st.number_input(
                f"Y{i+1}",
                value=float(poly[i][1]),
                step=2.0,
                format="%.1f",
                key=f"y_{gkey}_{i}",
                label_visibility="collapsed"
            )

        if new_x != poly[i][0] or new_y != poly[i][1]:
            new_poly[i] = (new_x, new_y)
            modified = True

    # Boutons de transformation
    st.markdown("---")
    st.markdown("**Transformations rapides:**")

    col1, col2 = st.columns(2)
    step = col1.slider("Pas", 2, 30, 10, key=f"step_{gkey}")
    scale_pct = col2.slider("Scale %", 80, 120, 100, 5, key=f"scale_{gkey}")

    # Boutons de déplacement
    c1, c2, c3, c4 = st.columns(4)
    move_up = c1.button("⬆️", key=f"up_{gkey}", use_container_width=True)
    move_down = c2.button("⬇️", key=f"dn_{gkey}", use_container_width=True)
    move_left = c3.button("⬅️", key=f"lt_{gkey}", use_container_width=True)
    move_right = c4.button("➡️", key=f"rt_{gkey}", use_container_width=True)

    # Boutons de scaling et rotation
    c5, c6, c7, c8 = st.columns(4)
    do_grow = c5.button("➕", key=f"grow_{gkey}", help="Agrandir", use_container_width=True)
    do_shrink = c6.button("➖", key=f"shrink_{gkey}", help="Réduire", use_container_width=True)
    do_rot_cw = c7.button("↻", key=f"rotcw_{gkey}", help="Rotation +15°", use_container_width=True)
    do_rot_ccw = c8.button("↺", key=f"rotccw_{gkey}", help="Rotation -15°", use_container_width=True)

    # Appliquer les transformations
    if modified:
        return new_poly
    if move_up:
        return move_polygon(poly, 0, step)
    if move_down:
        return move_polygon(poly, 0, -step)
    if move_left:
        return move_polygon(poly, -step, 0)
    if move_right:
        return move_polygon(poly, step, 0)
    if do_grow:
        return scale_polygon(poly, 1.1)
    if do_shrink:
        return scale_polygon(poly, 0.9)
    if do_rot_cw:
        return rotate_polygon(poly, 15)
    if do_rot_ccw:
        return rotate_polygon(poly, -15)
    if scale_pct != 100:
        return scale_polygon(poly, scale_pct / 100)

    return None


# ==================== MAIN APPLICATION ====================

st.markdown('<h1 class="main-header">🔬 FACS - Hexagones Interactifs v5</h1>', unsafe_allow_html=True)

# Chargement apprentissage
learned = load_learned_params()
n_learned = learned.get('n_corrections', 0)

if n_learned > 0:
    st.success(f"🧠 {n_learned} correction(s) apprises")

# Gestion de l'apprentissage
with st.expander("🧠 Gérer l'apprentissage", expanded=False):
    if n_learned == 0:
        st.info("Aucune correction enregistrée")
    else:
        st.markdown("**Corrections par gate:**")

        params = load_learned_params()
        gates_learned = params.get('gates', {})

        if gates_learned:
            for gate_name, gate_data in gates_learned.items():
                n_samples = gate_data.get('n_samples', 0)
                adj = gate_data.get('avg_adjustment', {'x': 0, 'y': 0})
                scale = gate_data.get('scale_factor', 1.0)
                conf_hist = gate_data.get('confidence_history', [])
                avg_conf = np.mean(conf_hist) if conf_hist else 0

                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                with col1:
                    st.write(f"**{gate_name}**: {n_samples} correction(s)")
                with col2:
                    st.caption(f"Δ=({adj['x']:.1f}, {adj['y']:.1f}), S={scale:.2f}")
                with col3:
                    if conf_hist:
                        conf_class, conf_icon = get_confidence_class(avg_conf)
                        st.markdown(f"{conf_icon} {avg_conf:.0f}%")
                with col4:
                    if st.button("🗑️", key=f"del_{gate_name}", help=f"Supprimer {gate_name}"):
                        del params['gates'][gate_name]
                        params['n_corrections'] = max(0, params['n_corrections'] - n_samples)
                        save_learned_params(params)
                        st.rerun()

        st.markdown("---")

        if st.button("🗑️ Réinitialiser tout l'apprentissage", type="secondary"):
            st.session_state.confirm_delete = True

        if st.session_state.get('confirm_delete', False):
            st.warning("⚠️ Cette action est irréversible.")
            col1, col2 = st.columns(2)
            if col1.button("✅ Confirmer", type="primary"):
                save_learned_params({'version': 2, 'n_corrections': 0, 'gates': {}})
                st.session_state.confirm_delete = False
                st.success("✅ Apprentissage réinitialisé")
                st.rerun()
            if col2.button("❌ Annuler"):
                st.session_state.confirm_delete = False
                st.rerun()

# Initialisation session state
for key in ['reader', 'data', 'channels', 'polygons', 'original_polygons',
            'auto_done', 'confidences', 'undo_stack']:
    if key not in st.session_state:
        st.session_state[key] = {} if key in ['channels', 'polygons', 'original_polygons', 'confidences'] else ([] if key == 'undo_stack' else None if key != 'auto_done' else False)

# Upload
uploaded = st.file_uploader("📁 Fichier FCS", type=['fcs'])

if uploaded:
    if st.session_state.reader is None or st.session_state.get('fname') != uploaded.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Chargement du fichier..."):
            reader = FCSReader(tmp_path)
            st.session_state.reader = reader
            st.session_state.data = reader.data
            st.session_state.fname = uploaded.name
            st.session_state.polygons = {}
            st.session_state.original_polygons = {}
            st.session_state.confidences = {}
            st.session_state.auto_done = False
            st.session_state.undo_stack = []

            cols = list(reader.data.columns)
            st.session_state.channels = {
                'FSC-A': find_channel(cols, ['FSC-A']),
                'FSC-H': find_channel(cols, ['FSC-H']),
                'SSC-A': find_channel(cols, ['SSC-A']),
                'LiveDead': find_channel(cols, ['LiveDead', 'Viab', 'Aqua', 'Live', 'L/D']),
                'hCD45': find_channel(cols, ['PerCP', 'CD45', 'hCD45']),
                'CD3': find_channel(cols, ['AF488', 'FITC', 'CD3']),
                'CD19': find_channel(cols, ['PE-Fire', 'CD19']),
                'CD4': find_channel(cols, ['BV650', 'CD4']),
                'CD8': find_channel(cols, ['BUV805', 'CD8']),
            }

    reader = st.session_state.reader
    data = st.session_state.data
    ch = st.session_state.channels
    n_total = len(data)

    # Métriques
    c1, c2, c3 = st.columns(3)
    c1.metric("Événements", f"{n_total:,}")
    c2.metric("Canaux", len(reader.channels))
    c3.metric("Fichier", reader.filename[:25])

    # Canaux détectés
    with st.expander("📋 Canaux détectés"):
        for name, canal in ch.items():
            st.write(f"{'✅' if canal else '❌'} **{name}**: {canal or 'Non trouvé'}")

    if ch['FSC-A'] is None or ch['SSC-A'] is None:
        st.error("❌ FSC-A ou SSC-A non trouvé!")
        st.stop()

    st.markdown("---")

    # Options d'auto-gating
    with st.expander("⚙️ Options d'auto-gating", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            remove_outliers_opt = st.checkbox("Supprimer outliers", value=True,
                                               help="Améliore la robustesse sur données bruitées")
        with col2:
            adaptive_comp = st.checkbox("Composantes adaptatives", value=True,
                                        help="Sélection automatique du nombre de clusters")

    # AUTO-GATING
    if not st.session_state.auto_done:
        if st.button("🚀 LANCER L'AUTO-GATING", type="primary", use_container_width=True):
            progress = st.progress(0, text="Initialisation...")

            with st.spinner("Auto-gating en cours..."):
                # 1. Cells
                progress.progress(10, text="Gate Cells...")
                poly, conf = auto_gate_hexagon_robust(
                    data, ch['FSC-A'], ch['SSC-A'], None, 'main',
                    remove_outliers_opt, adaptive_comp
                )
                poly = apply_learned_adj(poly, 'cells')
                st.session_state.polygons['cells'] = poly
                st.session_state.original_polygons['cells'] = list(poly) if poly else None
                st.session_state.confidences['cells'] = conf

                # 2. Singlets
                progress.progress(30, text="Gate Singlets...")
                if ch['FSC-H'] and poly:
                    cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], poly, None)
                    poly2, conf2 = auto_gate_hexagon_robust(
                        data, ch['FSC-A'], ch['FSC-H'], cells_m, 'main',
                        remove_outliers_opt, adaptive_comp
                    )
                    poly2 = apply_learned_adj(poly2, 'singlets')
                else:
                    poly2, conf2 = None, 0.0
                st.session_state.polygons['singlets'] = poly2
                st.session_state.original_polygons['singlets'] = list(poly2) if poly2 else None
                st.session_state.confidences['singlets'] = conf2

                # 3. Live
                progress.progress(55, text="Gate Live...")
                if ch['LiveDead']:
                    cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'],
                                        st.session_state.polygons['cells'], None)
                    sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'],
                                       st.session_state.polygons['singlets'], cells_m) \
                             if st.session_state.polygons['singlets'] else cells_m
                    poly3, conf3 = auto_gate_hexagon_robust(
                        data, ch['LiveDead'], ch['SSC-A'], sing_m, 'low_x',
                        remove_outliers_opt, adaptive_comp
                    )
                    poly3 = apply_learned_adj(poly3, 'live')
                else:
                    poly3, conf3 = None, 0.0
                st.session_state.polygons['live'] = poly3
                st.session_state.original_polygons['live'] = list(poly3) if poly3 else None
                st.session_state.confidences['live'] = conf3

                # 4. hCD45+
                progress.progress(80, text="Gate hCD45+...")
                if ch['hCD45']:
                    cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'],
                                        st.session_state.polygons['cells'], None)
                    sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'],
                                       st.session_state.polygons['singlets'], cells_m) \
                             if st.session_state.polygons['singlets'] else cells_m
                    live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'],
                                       st.session_state.polygons['live'], sing_m) \
                             if st.session_state.polygons['live'] else sing_m
                    poly4, conf4 = auto_gate_hexagon_robust(
                        data, ch['hCD45'], ch['SSC-A'], live_m, 'high_x',
                        remove_outliers_opt, adaptive_comp
                    )
                    poly4 = apply_learned_adj(poly4, 'hcd45')
                else:
                    poly4, conf4 = None, 0.0
                st.session_state.polygons['hcd45'] = poly4
                st.session_state.original_polygons['hcd45'] = list(poly4) if poly4 else None
                st.session_state.confidences['hcd45'] = conf4

                progress.progress(100, text="Terminé!")
                st.session_state.auto_done = True
                st.rerun()

    # AFFICHAGE
    if st.session_state.auto_done:
        polygons = st.session_state.polygons
        confidences = st.session_state.confidences

        # Recalcul cascade
        cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], polygons.get('cells'), None)
        sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], polygons.get('singlets'), cells_m) \
                 if polygons.get('singlets') else cells_m
        live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'], polygons.get('live'), sing_m) \
                 if polygons.get('live') else sing_m
        hcd45_m = apply_gate(data, ch['hCD45'], ch['SSC-A'], polygons.get('hcd45'), live_m) \
                  if polygons.get('hcd45') else live_m

        # Config
        gates_config = [
            ('cells', 'Cells', ch['FSC-A'], ch['SSC-A'], 'FSC-A', 'SSC-A', 'Ungated → Cells', None),
            ('singlets', 'Singlets', ch['FSC-A'], ch['FSC-H'], 'FSC-A', 'FSC-H', 'Cells → Singlets', cells_m),
            ('live', 'Live', ch['LiveDead'], ch['SSC-A'], 'Live/Dead', 'SSC-A', 'Singlets → Live', sing_m),
            ('hcd45', 'hCD45+', ch['hCD45'], ch['SSC-A'], 'hCD45', 'SSC-A', 'Live → hCD45+', live_m),
        ]

        stats = []

        # Affichage 2x2 avec édition améliorée
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
                    # En-tête avec confiance
                    conf = confidences.get(gkey, 0)
                    conf_class, conf_icon = get_confidence_class(conf)
                    st.markdown(f"#### {gname} {conf_icon}")

                    # Graphique
                    fig, n_in, pct = create_plot(
                        data, x_ch, y_ch, x_label, y_label, title,
                        polygons.get(gkey), parent_mask, gname, conf
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{gkey}")

                    parent_name = title.split('→')[0].strip()
                    stats.append((gname, parent_name, n_in, pct, conf))

                    # Éditeur de points
                    poly = polygons.get(gkey)
                    if poly:
                        with st.expander(f"✏️ Modifier {gname}", expanded=False):
                            new_poly = render_point_editor(gkey, poly, gname)
                            if new_poly is not None:
                                # Sauvegarder pour undo
                                st.session_state.undo_stack.append({
                                    'gate': gkey,
                                    'polygon': list(poly)
                                })
                                # Limiter la pile undo
                                st.session_state.undo_stack = st.session_state.undo_stack[-10:]
                                # Appliquer
                                st.session_state.polygons[gkey] = new_poly
                                # Recalculer confiance
                                new_conf = compute_gate_confidence(data, new_poly, x_ch, y_ch, parent_mask)
                                st.session_state.confidences[gkey] = new_conf
                                st.rerun()

        st.markdown("---")

        # Actions
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            if st.button("💾 Sauvegarder", type="primary", use_container_width=True):
                n_saved = 0
                for gname in polygons:
                    curr = polygons.get(gname)
                    orig = st.session_state.original_polygons.get(gname)
                    conf = confidences.get(gname)
                    if curr and orig and str(curr) != str(orig):
                        update_learned_params(gname, orig, curr, conf)
                        n_saved += 1
                if n_saved:
                    st.success(f"✅ {n_saved} correction(s) sauvegardées")
                else:
                    st.info("Aucune modification")

        with col_b:
            if st.button("↩️ Undo", use_container_width=True,
                        disabled=len(st.session_state.undo_stack) == 0):
                if st.session_state.undo_stack:
                    last = st.session_state.undo_stack.pop()
                    st.session_state.polygons[last['gate']] = last['polygon']
                    st.rerun()

        with col_c:
            if st.button("🔃 Reset", use_container_width=True):
                st.session_state.polygons = {
                    k: list(v) if v else None
                    for k, v in st.session_state.original_polygons.items()
                }
                st.session_state.undo_stack = []
                st.rerun()

        with col_d:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        # Résumé avec confiance
        st.markdown("### 📊 Résumé")
        df = pd.DataFrame(stats, columns=['Population', 'Parent', 'Count', '% Parent', 'Confiance'])
        df['% Total'] = (df['Count'] / n_total * 100).round(2)
        df['% Parent'] = df['% Parent'].round(1)
        df['Confiance'] = df['Confiance'].round(0).astype(int).astype(str) + '%'

        # Afficher avec style
        st.dataframe(
            df[['Population', 'Parent', 'Count', '% Parent', '% Total', 'Confiance']],
            use_container_width=True,
            hide_index=True
        )

        # Export
        c1, c2 = st.columns(2)
        export_df = df.drop('Confiance', axis=1)
        c1.download_button(
            "📥 CSV",
            export_df.to_csv(index=False),
            f"{reader.filename}_gating.csv",
            "text/csv",
            use_container_width=True
        )

        buf = io.BytesIO()
        export_df.to_excel(buf, index=False, engine='openpyxl')
        buf.seek(0)
        c2.download_button(
            "📥 Excel",
            buf,
            f"{reader.filename}_gating.xlsx",
            use_container_width=True
        )

else:
    st.markdown("""
    <div class="info-box">
    <h3>🔬 FACS Auto-Gating v5 - Améliorations</h3>
    <ul>
    <li>🎯 <b>GMM robuste</b>: Détection outliers, sélection BIC, initialisations multiples</li>
    <li>📊 <b>Score de confiance</b>: Évaluation automatique de la qualité des gates</li>
    <li>✏️ <b>Édition améliorée</b>: Tous les points visibles, rotation, undo</li>
    <li>🧠 <b>Apprentissage enrichi</b>: Translation, scaling, historique de confiance</li>
    </ul>
    <p>Uploadez un fichier FCS pour commencer</p>
    </div>
    """, unsafe_allow_html=True)

st.caption(f"v5 | 🧠 {n_learned} corrections | Optimisé pour fiabilité")
