#!/usr/bin/env python3
"""
FACS Autogating - Gates Polygonaux Interactifs
- Polygones de gating dessinables directement sur chaque graphe
- Dessinez vos propres gates avec l'outil polygon
- Workflow immunophénotypage complet
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import io
from datetime import datetime
import flowio
import re
import json

# Configuration
st.set_page_config(
    page_title="FACS - Gates Polygonaux",
    page_icon="🔬",
    layout="wide"
)

# CSS
st.markdown("""
    <style>
    .main-header { font-size: 2rem; color: #2c3e50; text-align: center; }
    .instruction { background: #e8f4f8; padding: 1rem; border-radius: 0.5rem;
                   border-left: 4px solid #3498db; margin: 0.5rem 0; }
    .gate-info { background: #f0fff0; padding: 0.5rem; border-radius: 0.3rem;
                 border-left: 3px solid #28a745; margin: 0.3rem 0; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)


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


def point_in_polygon(x, y, poly_x, poly_y):
    """
    Vérifie si les points (x, y) sont dans le polygone défini par (poly_x, poly_y)
    Utilise l'algorithme ray casting
    """
    n = len(poly_x)
    inside = np.zeros(len(x), dtype=bool)
    
    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        
        # Vérifier si le rayon horizontal croise le segment
        cond1 = (yi > y) != (yj > y)
        slope = (xj - xi) / (yj - yi + 1e-10)
        x_intersect = xi + slope * (y - yi)
        cond2 = x < x_intersect
        
        inside = inside ^ (cond1 & cond2)
        j = i
    
    return inside


def create_polygon_plot(data, x_channel, y_channel, x_marker, y_marker, 
                        title, polygon_coords=None, parent_mask=None,
                        gate_name="Gate", colorscale='Jet'):
    """
    Crée un graphique Plotly avec possibilité de dessiner un polygone
    """
    
    # Appliquer le masque parent
    if parent_mask is not None:
        plot_data = data[parent_mask].copy()
        parent_indices = data.index[parent_mask]
    else:
        plot_data = data.copy()
        parent_indices = data.index
    
    if len(plot_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig, pd.Series(False, index=data.index), 0, 0
    
    # Extraire les données
    x_data = plot_data[x_channel].values
    y_data = plot_data[y_channel].values
    
    valid = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
    x_valid = x_data[valid]
    y_valid = y_data[valid]
    valid_indices = parent_indices[valid]
    
    if len(x_valid) == 0:
        fig = go.Figure()
        return fig, pd.Series(False, index=data.index), 0, 0
    
    # Transformer pour affichage
    x_plot = biex_transform(x_valid)
    y_plot = biex_transform(y_valid)
    
    n_total = len(x_valid)
    
    # Sous-échantillonner si trop de points
    max_points = 15000
    if len(x_plot) > max_points:
        idx = np.random.choice(len(x_plot), max_points, replace=False)
        x_display = x_plot[idx]
        y_display = y_plot[idx]
    else:
        x_display = x_plot
        y_display = y_plot
    
    # Créer la figure
    fig = go.Figure()
    
    # Points colorés par densité (style FlowJo)
    fig.add_trace(go.Scattergl(
        x=x_display, 
        y=y_display,
        mode='markers',
        marker=dict(
            size=3,
            color=y_display,
            colorscale=colorscale,
            opacity=0.7,
            showscale=False
        ),
        name='Events',
        hoverinfo='skip'
    ))
    
    # Calculer les événements dans le gate
    n_in_gate = 0
    pct = 0
    gate_mask = pd.Series(False, index=data.index)
    
    if polygon_coords and len(polygon_coords) >= 3:
        # Extraire les coordonnées du polygone
        poly_x = [p[0] for p in polygon_coords]
        poly_y = [p[1] for p in polygon_coords]
        
        # Fermer le polygone
        if poly_x[0] != poly_x[-1] or poly_y[0] != poly_y[-1]:
            poly_x.append(poly_x[0])
            poly_y.append(poly_y[0])
        
        # Dessiner le polygone
        fig.add_trace(go.Scatter(
            x=poly_x, y=poly_y,
            mode='lines',
            line=dict(color='red', width=3),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.15)',
            name=gate_name
        ))
        
        # Calculer quels points sont dans le polygone (en coordonnées transformées)
        in_poly = point_in_polygon(x_plot, y_plot, np.array(poly_x), np.array(poly_y))
        n_in_gate = in_poly.sum()
        pct = n_in_gate / n_total * 100 if n_total > 0 else 0
        
        # Créer le masque complet
        gate_mask.loc[valid_indices[in_poly]] = True
        
        # Annotation
        centroid_x = np.mean(poly_x[:-1])
        centroid_y = np.mean(poly_y[:-1])
        
        fig.add_annotation(
            x=centroid_x, y=centroid_y,
            text=f"<b>{gate_name}</b><br>{pct:.1f}%<br>({n_in_gate:,})",
            showarrow=False,
            font=dict(size=11, color="darkred"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="red",
            borderwidth=2,
            borderpad=4
        )
    
    # Layout avec fond blanc
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sup>n={n_total:,}</sup>", x=0.5, font=dict(size=14)),
        xaxis_title=f"<b>{x_marker}</b>",
        yaxis_title=f"<b>{y_marker}</b>",
        showlegend=False,
        width=500,
        height=450,
        dragmode='drawclosedpath',  # Mode dessin de polygone par défaut
        margin=dict(l=60, r=20, t=70, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor='black',
            linewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor='black',
            linewidth=1
        ),
        newshape=dict(
            line=dict(color='red', width=3),
            fillcolor='rgba(255, 0, 0, 0.15)'
        ),
        modebar=dict(
            add=['drawclosedpath', 'eraseshape'],
            remove=['lasso2d', 'select2d', 'drawrect', 'drawline', 'drawcircle']
        )
    )
    
    # Ajouter boutons pour le mode de dessin
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.15,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        args=[{"dragmode": "drawclosedpath"}],
                        label="✏️ Dessiner Gate",
                        method="relayout"
                    ),
                    dict(
                        args=[{"dragmode": "pan"}],
                        label="🖐️ Déplacer",
                        method="relayout"
                    ),
                    dict(
                        args=[{"dragmode": "zoom"}],
                        label="🔍 Zoom",
                        method="relayout"
                    ),
                ]
            )
        ]
    )
    
    return fig, gate_mask, n_in_gate, pct


def create_default_polygon(data, x_channel, y_channel, percentiles=(10, 90)):
    """Crée un polygone par défaut basé sur les percentiles des données"""
    x_data = data[x_channel].values
    y_data = data[y_channel].values
    
    valid = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
    x_valid = x_data[valid]
    y_valid = y_data[valid]
    
    if len(x_valid) == 0:
        return []
    
    x_plot = biex_transform(x_valid)
    y_plot = biex_transform(y_valid)
    
    x_min, x_max = np.percentile(x_plot, percentiles)
    y_min, y_max = np.percentile(y_plot, percentiles)
    
    # Créer un polygone hexagonal (plus naturel pour FACS)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    rx = (x_max - x_min) / 2 * 0.9
    ry = (y_max - y_min) / 2 * 0.9
    
    # Points du polygone (forme elliptique approximée)
    angles = np.linspace(0, 2*np.pi, 8)[:-1]  # 7 points
    polygon = []
    for a in angles:
        px = cx + rx * np.cos(a)
        py = cy + ry * np.sin(a)
        polygon.append([float(px), float(py)])
    
    return polygon


# ==================== INTERFACE STREAMLIT ====================

st.markdown('<h1 class="main-header">🔬 FACS - Gates Polygonaux</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="instruction">
<b>📌 Instructions :</b><br>
1. Cliquez sur <b>"✏️ Dessiner Gate"</b> en haut à gauche de chaque graphique<br>
2. <b>Cliquez</b> sur le graphique pour placer les points du polygone<br>
3. <b>Double-cliquez</b> pour fermer le polygone<br>
4. Utilisez <b>"Appliquer les gates"</b> pour calculer les statistiques
</div>
""", unsafe_allow_html=True)

# Session state
if 'polygons' not in st.session_state:
    st.session_state.polygons = {}
if 'masks' not in st.session_state:
    st.session_state.masks = {}
if 'reader' not in st.session_state:
    st.session_state.reader = None

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
                st.session_state.masks = {}
        
        reader = st.session_state.reader
        data = reader.data
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        col1.metric("Événements", f"{len(data):,}")
        col2.metric("Canaux", len(reader.channels))
        col3.metric("Fichier", reader.filename[:25])
        
        # Trouver les canaux
        fsc_a = find_channel(data, ['FSC-A'])
        fsc_h = find_channel(data, ['FSC-H'])
        ssc_a = find_channel(data, ['SSC-A'])
        livedead = find_channel(data, ['LiveDead', 'Viab'])
        hcd45 = find_channel(data, ['PerCP-A'])
        cd3 = find_channel(data, ['AF488', 'CD3'])
        cd19 = find_channel(data, ['PE-Fire700', 'CD19'])
        cd4 = find_channel(data, ['BV650', 'CD4'])
        cd8 = find_channel(data, ['BUV805', 'CD8'])
        cd56 = find_channel(data, ['PE-Cy7', 'CD56'])
        cd16 = find_channel(data, ['NovaFluor', 'CD16'])
        foxp3 = find_channel(data, ['eFluor450', 'FoxP3'])
        cd25 = find_channel(data, ['BV785', 'CD25'])
        
        # Initialiser les polygones par défaut
        if 'cells' not in st.session_state.polygons and fsc_a and ssc_a:
            st.session_state.polygons['cells'] = create_default_polygon(data, fsc_a, ssc_a, (5, 95))
        if 'singlets' not in st.session_state.polygons and fsc_a and fsc_h:
            st.session_state.polygons['singlets'] = create_default_polygon(data, fsc_a, fsc_h, (3, 97))
        if 'live' not in st.session_state.polygons and livedead and ssc_a:
            st.session_state.polygons['live'] = create_default_polygon(data, livedead, ssc_a, (0, 85))
        if 'hcd45' not in st.session_state.polygons and hcd45 and ssc_a:
            st.session_state.polygons['hcd45'] = create_default_polygon(data, hcd45, ssc_a, (15, 99))
        if 'tb' not in st.session_state.polygons and cd3 and cd19:
            st.session_state.polygons['tb'] = create_default_polygon(data, cd3, cd19, (40, 99))
        if 'cd4cd8' not in st.session_state.polygons and cd4 and cd8:
            st.session_state.polygons['cd4cd8'] = create_default_polygon(data, cd4, cd8, (35, 99))
        
        st.markdown("---")
        
        # ==================== GATING ====================
        
        all_stats = []
        n_total = len(data)
        
        # ===== ROW 1 =====
        st.markdown("### 1️⃣ Gating Principal")
        
        col1, col2 = st.columns(2)
        
        # Plot 1: Cells
        with col1:
            st.markdown("##### 🔹 Cells (FSC-A vs SSC-A)")
            if fsc_a and ssc_a:
                fig1, mask1, n1, pct1 = create_polygon_plot(
                    data, fsc_a, ssc_a, 'FSC-A', 'SSC-A',
                    f"Ungated (n={n_total:,})",
                    polygon_coords=st.session_state.polygons.get('cells'),
                    gate_name="Cells"
                )
                
                # Capture du polygone dessiné
                plot1 = st.plotly_chart(fig1, use_container_width=True, key="cells_plot", 
                                        on_select="ignore")
                
                # Input pour les coordonnées du polygone
                with st.expander("📐 Coordonnées du polygone (Cells)"):
                    coords_str = st.text_area(
                        "Format: x1,y1;x2,y2;x3,y3;...",
                        value=";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in st.session_state.polygons.get('cells', [])]),
                        key="cells_coords",
                        height=68
                    )
                    if coords_str:
                        try:
                            new_coords = [[float(c) for c in p.split(',')] for p in coords_str.split(';') if p.strip()]
                            if len(new_coords) >= 3:
                                st.session_state.polygons['cells'] = new_coords
                        except:
                            pass
                
                st.session_state.masks['cells'] = mask1
                if mask1.sum() > 0:
                    st.markdown(f'<div class="gate-info">✅ <b>Cells:</b> {mask1.sum():,} ({mask1.sum()/n_total*100:.1f}%)</div>', 
                               unsafe_allow_html=True)
                    all_stats.append({'Population': 'Cells', 'Parent': 'Ungated',
                                     'Count': mask1.sum(), '% Parent': round(mask1.sum()/n_total*100, 1)})
        
        # Plot 2: Singlets
        with col2:
            st.markdown("##### 🔹 Single Cells (FSC-A vs FSC-H)")
            if fsc_a and fsc_h:
                parent = st.session_state.masks.get('cells')
                n_parent = parent.sum() if parent is not None else n_total
                
                fig2, mask2, n2, pct2 = create_polygon_plot(
                    data, fsc_a, fsc_h, 'FSC-A', 'FSC-H',
                    f"Cells (n={n_parent:,})",
                    polygon_coords=st.session_state.polygons.get('singlets'),
                    parent_mask=parent,
                    gate_name="Single Cells"
                )
                st.plotly_chart(fig2, use_container_width=True, key="singlets_plot")
                
                with st.expander("📐 Coordonnées du polygone (Singlets)"):
                    coords_str = st.text_area(
                        "Format: x1,y1;x2,y2;...",
                        value=";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in st.session_state.polygons.get('singlets', [])]),
                        key="singlets_coords",
                        height=68
                    )
                    if coords_str:
                        try:
                            new_coords = [[float(c) for c in p.split(',')] for p in coords_str.split(';') if p.strip()]
                            if len(new_coords) >= 3:
                                st.session_state.polygons['singlets'] = new_coords
                        except:
                            pass
                
                st.session_state.masks['singlets'] = mask2
                if mask2.sum() > 0:
                    st.markdown(f'<div class="gate-info">✅ <b>Singlets:</b> {mask2.sum():,} ({mask2.sum()/n_parent*100:.1f}%)</div>',
                               unsafe_allow_html=True)
                    all_stats.append({'Population': 'Single Cells', 'Parent': 'Cells',
                                     'Count': mask2.sum(), '% Parent': round(mask2.sum()/n_parent*100, 1)})
        
        # ===== ROW 2 =====
        col3, col4 = st.columns(2)
        
        # Plot 3: Live
        with col3:
            st.markdown("##### 🔹 Live (Live/Dead vs SSC-A)")
            if livedead and ssc_a:
                parent = st.session_state.masks.get('singlets')
                n_parent = parent.sum() if parent is not None else n_total
                
                fig3, mask3, n3, pct3 = create_polygon_plot(
                    data, livedead, ssc_a, 'Live/Dead', 'SSC-A',
                    f"Singlets (n={n_parent:,})",
                    polygon_coords=st.session_state.polygons.get('live'),
                    parent_mask=parent,
                    gate_name="Live"
                )
                st.plotly_chart(fig3, use_container_width=True, key="live_plot")
                
                with st.expander("📐 Coordonnées du polygone (Live)"):
                    coords_str = st.text_area(
                        "Format: x1,y1;x2,y2;...",
                        value=";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in st.session_state.polygons.get('live', [])]),
                        key="live_coords",
                        height=68
                    )
                    if coords_str:
                        try:
                            new_coords = [[float(c) for c in p.split(',')] for p in coords_str.split(';') if p.strip()]
                            if len(new_coords) >= 3:
                                st.session_state.polygons['live'] = new_coords
                        except:
                            pass
                
                st.session_state.masks['live'] = mask3
                if mask3.sum() > 0:
                    st.markdown(f'<div class="gate-info">✅ <b>Live:</b> {mask3.sum():,} ({mask3.sum()/n_parent*100:.1f}%)</div>',
                               unsafe_allow_html=True)
                    all_stats.append({'Population': 'Live', 'Parent': 'Single Cells',
                                     'Count': mask3.sum(), '% Parent': round(mask3.sum()/n_parent*100, 1)})
        
        # Plot 4: hCD45+
        with col4:
            st.markdown("##### 🔹 hCD45+ (hCD45 vs SSC-A)")
            if hcd45 and ssc_a:
                parent = st.session_state.masks.get('live')
                n_parent = parent.sum() if parent is not None else n_total
                
                fig4, mask4, n4, pct4 = create_polygon_plot(
                    data, hcd45, ssc_a, 'hCD45', 'SSC-A',
                    f"Live (n={n_parent:,})",
                    polygon_coords=st.session_state.polygons.get('hcd45'),
                    parent_mask=parent,
                    gate_name="hCD45+"
                )
                st.plotly_chart(fig4, use_container_width=True, key="hcd45_plot")
                
                with st.expander("📐 Coordonnées du polygone (hCD45+)"):
                    coords_str = st.text_area(
                        "Format: x1,y1;x2,y2;...",
                        value=";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in st.session_state.polygons.get('hcd45', [])]),
                        key="hcd45_coords",
                        height=68
                    )
                    if coords_str:
                        try:
                            new_coords = [[float(c) for c in p.split(',')] for p in coords_str.split(';') if p.strip()]
                            if len(new_coords) >= 3:
                                st.session_state.polygons['hcd45'] = new_coords
                        except:
                            pass
                
                st.session_state.masks['hcd45'] = mask4
                if mask4.sum() > 0:
                    st.markdown(f'<div class="gate-info">✅ <b>hCD45+:</b> {mask4.sum():,} ({mask4.sum()/n_parent*100:.1f}%)</div>',
                               unsafe_allow_html=True)
                    all_stats.append({'Population': 'hCD45+ (Leucocytes)', 'Parent': 'Live',
                                     'Count': mask4.sum(), '% Parent': round(mask4.sum()/n_parent*100, 1)})
        
        # ===== ROW 3: SOUS-POPULATIONS =====
        st.markdown("### 2️⃣ Sous-Populations")
        
        col5, col6 = st.columns(2)
        
        # Plot 5: T cells (CD3 vs CD19)
        with col5:
            st.markdown("##### 🔹 T cells (CD3+ CD19-)")
            if cd3 and cd19:
                parent = st.session_state.masks.get('hcd45')
                n_parent = parent.sum() if parent is not None else n_total
                
                fig5, mask5, n5, pct5 = create_polygon_plot(
                    data, cd3, cd19, 'CD3', 'CD19',
                    f"hCD45+ (n={n_parent:,})",
                    polygon_coords=st.session_state.polygons.get('tb'),
                    parent_mask=parent,
                    gate_name="T cells"
                )
                st.plotly_chart(fig5, use_container_width=True, key="tb_plot")
                
                with st.expander("📐 Coordonnées du polygone (T cells)"):
                    coords_str = st.text_area(
                        "Format: x1,y1;x2,y2;...",
                        value=";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in st.session_state.polygons.get('tb', [])]),
                        key="tb_coords",
                        height=68
                    )
                    if coords_str:
                        try:
                            new_coords = [[float(c) for c in p.split(',')] for p in coords_str.split(';') if p.strip()]
                            if len(new_coords) >= 3:
                                st.session_state.polygons['tb'] = new_coords
                        except:
                            pass
                
                st.session_state.masks['t_cells'] = mask5
                if mask5.sum() > 0:
                    st.markdown(f'<div class="gate-info">✅ <b>T cells:</b> {mask5.sum():,} ({mask5.sum()/n_parent*100:.1f}%)</div>',
                               unsafe_allow_html=True)
                    all_stats.append({'Population': 'T cells', 'Parent': 'hCD45+',
                                     'Count': mask5.sum(), '% Parent': round(mask5.sum()/n_parent*100, 1)})
        
        # Plot 6: CD4/CD8
        with col6:
            st.markdown("##### 🔹 CD4+ T cells")
            if cd4 and cd8:
                parent = st.session_state.masks.get('t_cells')
                n_parent = parent.sum() if parent is not None else 0
                
                if n_parent > 0:
                    fig6, mask6, n6, pct6 = create_polygon_plot(
                        data, cd4, cd8, 'CD4', 'CD8',
                        f"T cells (n={n_parent:,})",
                        polygon_coords=st.session_state.polygons.get('cd4cd8'),
                        parent_mask=parent,
                        gate_name="CD4+"
                    )
                    st.plotly_chart(fig6, use_container_width=True, key="cd4cd8_plot")
                    
                    with st.expander("📐 Coordonnées du polygone (CD4+)"):
                        coords_str = st.text_area(
                            "Format: x1,y1;x2,y2;...",
                            value=";".join([f"{p[0]:.1f},{p[1]:.1f}" for p in st.session_state.polygons.get('cd4cd8', [])]),
                            key="cd4cd8_coords",
                            height=68
                        )
                        if coords_str:
                            try:
                                new_coords = [[float(c) for c in p.split(',')] for p in coords_str.split(';') if p.strip()]
                                if len(new_coords) >= 3:
                                    st.session_state.polygons['cd4cd8'] = new_coords
                            except:
                                pass
                    
                    st.session_state.masks['cd4_cells'] = mask6
                    if mask6.sum() > 0:
                        st.markdown(f'<div class="gate-info">✅ <b>CD4+ T:</b> {mask6.sum():,} ({mask6.sum()/n_parent*100:.1f}%)</div>',
                                   unsafe_allow_html=True)
                        all_stats.append({'Population': 'CD4+ T cells', 'Parent': 'T cells',
                                         'Count': mask6.sum(), '% Parent': round(mask6.sum()/n_parent*100, 1)})
                else:
                    st.info("Dessinez d'abord le gate T cells")
        
        # ==================== RECALCUL ====================
        
        st.markdown("---")
        
        if st.button("🔄 **Recalculer les statistiques**", type="primary", use_container_width=True):
            st.rerun()
        
        # ==================== STATISTIQUES ====================
        
        st.markdown("### 📊 Résumé des Populations")
        
        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_df['% Total'] = (stats_df['Count'] / n_total * 100).round(2)
            
            st.dataframe(stats_df, use_container_width=True, height=300)
            
            # Export
            col1, col2 = st.columns(2)
            
            with col1:
                csv = stats_df.to_csv(index=False)
                st.download_button(
                    "📥 Télécharger CSV",
                    csv,
                    f"{reader.filename}_populations.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                try:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        stats_df.to_excel(writer, sheet_name='Populations', index=False)
                        
                        # Ajouter feuille avec coordonnées des polygones
                        poly_data = []
                        for name, coords in st.session_state.polygons.items():
                            poly_data.append({
                                'Gate': name,
                                'Coordinates': ";".join([f"{p[0]:.2f},{p[1]:.2f}" for p in coords])
                            })
                        pd.DataFrame(poly_data).to_excel(writer, sheet_name='Gate_Coordinates', index=False)
                    
                    output.seek(0)
                    st.download_button(
                        "📥 Télécharger Excel",
                        output,
                        f"{reader.filename}_analysis.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erreur export Excel: {e}")
    
    except Exception as e:
        st.error(f"Erreur: {e}")
        st.exception(e)

else:
    st.info("👆 Uploadez un fichier FCS pour commencer")
    
    st.markdown("""
    ### 📌 Comment utiliser cette application
    
    1. **Uploadez** votre fichier FCS
    2. **Dessinez les gates polygonaux** :
       - Cliquez sur **"✏️ Dessiner Gate"** en haut de chaque graphique
       - **Cliquez** pour placer les points du polygone
       - **Double-cliquez** pour fermer le polygone
    3. **Modifiez** les coordonnées dans les expanders si besoin
    4. **Cliquez "Recalculer"** pour mettre à jour les statistiques
    5. **Exportez** vos résultats
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🔬 <b>FACS Analysis - Gates Polygonaux</b><br>
    Dessinez vos gates directement sur les graphiques
</div>
""", unsafe_allow_html=True)
