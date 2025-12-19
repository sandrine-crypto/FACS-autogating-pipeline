#!/usr/bin/env python3
"""
FACS Autogating - Gates Interactifs sur Graphiques
- Rectangles de gating visibles et ajustables directement sur chaque graphe
- Drag & drop pour déplacer/redimensionner les gates
- Mise à jour en temps réel des statistiques
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import tempfile
import io
from datetime import datetime
import flowio
import re

# Configuration
st.set_page_config(
    page_title="FACS - Gates Interactifs",
    page_icon="🔬",
    layout="wide"
)

# CSS
st.markdown("""
    <style>
    .main-header { font-size: 2rem; color: #2c3e50; text-align: center; }
    .stats-box { background: #f0f7ff; padding: 1rem; border-radius: 0.5rem; 
                 border-left: 4px solid #3498db; margin: 0.5rem 0; }
    .instruction { background: #fff3cd; padding: 0.8rem; border-radius: 0.5rem;
                   border-left: 4px solid #ffc107; margin: 0.5rem 0; font-size: 0.9rem; }
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


def create_interactive_plot(data, x_channel, y_channel, x_marker, y_marker, 
                            title, gate_coords=None, parent_mask=None,
                            plot_id="plot", is_quadrant=False):
    """
    Crée un graphique Plotly interactif avec gate ajustable
    gate_coords: dict avec x0, x1, y0, y1 pour rectangle ou x_thresh, y_thresh pour quadrant
    """
    
    # Appliquer le masque parent si fourni
    if parent_mask is not None:
        plot_data = data[parent_mask].copy()
    else:
        plot_data = data.copy()
    
    if len(plot_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig, 0, 0
    
    # Extraire et transformer les données
    x_data = plot_data[x_channel].values
    y_data = plot_data[y_channel].values
    
    valid = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
    x_valid = x_data[valid]
    y_valid = y_data[valid]
    
    if len(x_valid) == 0:
        fig = go.Figure()
        return fig, 0, 0
    
    # Transformer pour affichage
    x_plot = biex_transform(x_valid)
    y_plot = biex_transform(y_valid)
    
    # Sous-échantillonner si trop de points
    max_points = 20000
    if len(x_plot) > max_points:
        idx = np.random.choice(len(x_plot), max_points, replace=False)
        x_plot_display = x_plot[idx]
        y_plot_display = y_plot[idx]
    else:
        x_plot_display = x_plot
        y_plot_display = y_plot
    
    # Créer la figure
    fig = go.Figure()
    
    # Ajouter le scatter plot (density-like avec histogram2dcontour)
    fig.add_trace(go.Histogram2dContour(
        x=x_plot_display, y=y_plot_display,
        colorscale='Jet',
        showscale=False,
        contours=dict(showlines=False),
        ncontours=20,
        name='Density'
    ))
    
    # Ajouter les points en overlay
    fig.add_trace(go.Scattergl(
        x=x_plot_display, y=y_plot_display,
        mode='markers',
        marker=dict(size=2, color='rgba(0,0,100,0.3)'),
        name='Events',
        hoverinfo='skip'
    ))
    
    n_in_gate = 0
    pct = 0
    n_total = len(x_valid)
    
    if is_quadrant and gate_coords:
        # Mode quadrant avec lignes croisées
        x_thresh = gate_coords.get('x_thresh', np.median(x_valid))
        y_thresh = gate_coords.get('y_thresh', np.median(y_valid))
        
        x_thresh_t = biex_transform([x_thresh])[0]
        y_thresh_t = biex_transform([y_thresh])[0]
        
        # Lignes de quadrant (shapes editables)
        fig.add_shape(
            type="line",
            x0=x_thresh_t, x1=x_thresh_t,
            y0=min(y_plot), y1=max(y_plot),
            line=dict(color="red", width=3),
            editable=True,
            name="X threshold"
        )
        fig.add_shape(
            type="line",
            x0=min(x_plot), x1=max(x_plot),
            y0=y_thresh_t, y1=y_thresh_t,
            line=dict(color="red", width=3),
            editable=True,
            name="Y threshold"
        )
        
        # Calculer % dans chaque quadrant
        q1 = ((x_valid >= x_thresh) & (y_valid >= y_thresh)).sum()  # ++
        q2 = ((x_valid < x_thresh) & (y_valid >= y_thresh)).sum()   # -+
        q3 = ((x_valid < x_thresh) & (y_valid < y_thresh)).sum()    # --
        q4 = ((x_valid >= x_thresh) & (y_valid < y_thresh)).sum()   # +-
        
        # Annotations pour chaque quadrant
        x_range = max(x_plot) - min(x_plot)
        y_range = max(y_plot) - min(y_plot)
        
        annotations_pos = [
            (max(x_plot) - x_range*0.15, max(y_plot) - y_range*0.05, f"{q1/n_total*100:.1f}%"),  # Q1
            (min(x_plot) + x_range*0.05, max(y_plot) - y_range*0.05, f"{q2/n_total*100:.1f}%"),  # Q2
            (min(x_plot) + x_range*0.05, min(y_plot) + y_range*0.05, f"{q3/n_total*100:.1f}%"),  # Q3
            (max(x_plot) - x_range*0.15, min(y_plot) + y_range*0.05, f"{q4/n_total*100:.1f}%"),  # Q4
        ]
        
        for x_pos, y_pos, text in annotations_pos:
            fig.add_annotation(
                x=x_pos, y=y_pos, text=text,
                showarrow=False, font=dict(size=14, color="red", family="Arial Black"),
                bgcolor="white", bordercolor="red", borderwidth=1
            )
        
        n_in_gate = q1 + q4  # Exemple: positifs en X
        pct = n_in_gate / n_total * 100 if n_total > 0 else 0
        
    elif gate_coords:
        # Mode rectangle
        x0 = gate_coords.get('x0', np.percentile(x_valid, 5))
        x1 = gate_coords.get('x1', np.percentile(x_valid, 95))
        y0 = gate_coords.get('y0', np.percentile(y_valid, 5))
        y1 = gate_coords.get('y1', np.percentile(y_valid, 95))
        
        # Transformer les coordonnées
        x0_t = biex_transform([x0])[0]
        x1_t = biex_transform([x1])[0]
        y0_t = biex_transform([y0])[0]
        y1_t = biex_transform([y1])[0]
        
        # Rectangle editable
        fig.add_shape(
            type="rect",
            x0=x0_t, x1=x1_t, y0=y0_t, y1=y1_t,
            line=dict(color="red", width=3),
            fillcolor="rgba(255,0,0,0.1)",
            editable=True,
            name="Gate"
        )
        
        # Calculer % dans le gate
        in_gate = (x_valid >= x0) & (x_valid <= x1) & (y_valid >= y0) & (y_valid <= y1)
        n_in_gate = in_gate.sum()
        pct = n_in_gate / n_total * 100 if n_total > 0 else 0
        
        # Annotation avec stats
        fig.add_annotation(
            x=x1_t, y=y1_t,
            text=f"<b>{gate_coords.get('name', 'Gate')}</b><br>{pct:.1f}%<br>({n_in_gate:,})",
            showarrow=True, arrowhead=2,
            font=dict(size=12, color="red"),
            bgcolor="white", bordercolor="red", borderwidth=2,
            ax=30, ay=-30
        )
    
    # Configuration du layout
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sup>n={n_total:,}</sup>", x=0.5),
        xaxis_title=f"<b>{x_marker}</b>",
        yaxis_title=f"<b>{y_marker}</b>",
        showlegend=False,
        width=450,
        height=400,
        dragmode='pan',
        margin=dict(l=60, r=20, t=60, b=60),
    )
    
    # Permettre l'édition des shapes
    fig.update_layout(
        newshape=dict(line_color='red', fillcolor='rgba(255,0,0,0.1)'),
        modebar=dict(
            add=['drawrect', 'drawline', 'eraseshape'],
            remove=['lasso2d', 'select2d']
        )
    )
    
    return fig, n_in_gate, pct


def apply_gate(data, x_channel, y_channel, gate_coords, parent_mask=None):
    """Applique un gate et retourne le masque"""
    if parent_mask is not None:
        working_mask = parent_mask.copy()
    else:
        working_mask = pd.Series(True, index=data.index)
    
    x_data = data[x_channel]
    y_data = data[y_channel]
    
    if 'x0' in gate_coords:
        # Rectangle gate
        gate_mask = (x_data >= gate_coords['x0']) & (x_data <= gate_coords['x1']) & \
                    (y_data >= gate_coords['y0']) & (y_data <= gate_coords['y1'])
    else:
        # Quadrant gate (retourne le quadrant positif-positif par défaut)
        gate_mask = (x_data >= gate_coords.get('x_thresh', 0)) & \
                    (y_data >= gate_coords.get('y_thresh', 0))
    
    return working_mask & gate_mask


# ==================== INTERFACE STREAMLIT ====================

st.markdown('<h1 class="main-header">🔬 FACS - Gates Interactifs sur Graphiques</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="instruction">
📌 <b>Instructions:</b> Les rectangles rouges sont les gates. 
<b>Cliquez et glissez</b> les bords ou coins pour ajuster chaque gate directement sur le graphique.
Les lignes rouges dans les quadrants peuvent aussi être déplacées.
</div>
""", unsafe_allow_html=True)

# Session state
if 'gates' not in st.session_state:
    st.session_state.gates = {}
if 'masks' not in st.session_state:
    st.session_state.masks = {}

# Upload
uploaded_file = st.file_uploader("📁 Fichier FCS", type=['fcs'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    try:
        with st.spinner("Chargement..."):
            reader = FCSReader(tmp_path)
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
        mcd45 = find_channel(data, ['APC-Fire750'])
        cd3 = find_channel(data, ['AF488', 'CD3'])
        cd19 = find_channel(data, ['PE-Fire700', 'CD19'])
        cd4 = find_channel(data, ['BV650', 'CD4'])
        cd8 = find_channel(data, ['BUV805', 'CD8'])
        cd56 = find_channel(data, ['PE-Cy7', 'CD56'])
        cd16 = find_channel(data, ['NovaFluor', 'CD16'])
        foxp3 = find_channel(data, ['eFluor450', 'FoxP3'])
        cd25 = find_channel(data, ['BV785', 'CD25'])
        
        # Initialiser les gates avec valeurs par défaut
        if 'initialized' not in st.session_state:
            st.session_state.gates = {
                'cells': {
                    'x0': float(np.percentile(data[fsc_a], 3)) if fsc_a else 0,
                    'x1': float(np.percentile(data[fsc_a], 99)) if fsc_a else 1,
                    'y0': float(np.percentile(data[ssc_a], 3)) if ssc_a else 0,
                    'y1': float(np.percentile(data[ssc_a], 99)) if ssc_a else 1,
                    'name': 'Cells'
                },
                'singlets': {
                    'x0': float(np.percentile(data[fsc_a], 2)) if fsc_a else 0,
                    'x1': float(np.percentile(data[fsc_a], 98)) if fsc_a else 1,
                    'y0': float(np.percentile(data[fsc_h], 2)) if fsc_h else 0,
                    'y1': float(np.percentile(data[fsc_h], 98)) if fsc_h else 1,
                    'name': 'Single Cells'
                },
                'live': {
                    'x0': float(np.percentile(data[livedead], 0)) if livedead else 0,
                    'x1': float(np.percentile(data[livedead], 85)) if livedead else 1,
                    'y0': float(np.percentile(data[ssc_a], 2)) if ssc_a else 0,
                    'y1': float(np.percentile(data[ssc_a], 98)) if ssc_a else 1,
                    'name': 'Live'
                },
                'hcd45': {
                    'x0': float(np.percentile(data[hcd45], 15)) if hcd45 else 0,
                    'x1': float(np.percentile(data[hcd45], 100)) if hcd45 else 1,
                    'y0': float(np.percentile(data[ssc_a], 2)) if ssc_a else 0,
                    'y1': float(np.percentile(data[ssc_a], 98)) if ssc_a else 1,
                    'name': 'hCD45+'
                },
                'nk': {
                    'x_thresh': float(np.percentile(data[cd56], 65)) if cd56 else 0,
                    'y_thresh': float(np.percentile(data[cd16], 65)) if cd16 else 0,
                },
                'tb': {
                    'x_thresh': float(np.percentile(data[cd3], 40)) if cd3 else 0,
                    'y_thresh': float(np.percentile(data[cd19], 75)) if cd19 else 0,
                },
                'cd4cd8': {
                    'x_thresh': float(np.percentile(data[cd4], 35)) if cd4 else 0,
                    'y_thresh': float(np.percentile(data[cd8], 75)) if cd8 else 0,
                },
                'treg': {
                    'x_thresh': float(np.percentile(data[foxp3], 90)) if foxp3 else 0,
                    'y_thresh': float(np.percentile(data[cd25], 85)) if cd25 else 0,
                },
            }
            st.session_state.initialized = True
        
        st.markdown("---")
        
        # ==================== GATING HIÉRARCHIQUE ====================
        
        st.markdown("## 🔬 Gating Hiérarchique")
        st.markdown("*Ajustez les rectangles rouges directement sur chaque graphique*")
        
        all_stats = []
        n_total = len(data)
        
        # ===== ROW 1: GATING PRINCIPAL =====
        st.markdown("### 1️⃣ Gating Principal")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Plot 1: Cells
        with col1:
            st.markdown("##### Cells (FSC-A vs SSC-A)")
            if fsc_a and ssc_a:
                fig1, n_cells, pct_cells = create_interactive_plot(
                    data, fsc_a, ssc_a, 'FSC-A', 'SSC-A',
                    f"Ungated (n={n_total:,})",
                    gate_coords=st.session_state.gates['cells']
                )
                st.plotly_chart(fig1, use_container_width=True, key="plot_cells")
                
                cells_mask = apply_gate(data, fsc_a, ssc_a, st.session_state.gates['cells'])
                st.session_state.masks['cells'] = cells_mask
                
                st.markdown(f"**Cells:** {cells_mask.sum():,} ({cells_mask.sum()/n_total*100:.1f}%)")
                all_stats.append({'Population': 'Cells', 'Parent': 'Ungated', 
                                 'Count': cells_mask.sum(), '% Parent': round(cells_mask.sum()/n_total*100, 1)})
        
        # Plot 2: Singlets
        with col2:
            st.markdown("##### Single Cells (FSC-A vs FSC-H)")
            if fsc_a and fsc_h and 'cells' in st.session_state.masks:
                fig2, n_sing, pct_sing = create_interactive_plot(
                    data, fsc_a, fsc_h, 'FSC-A', 'FSC-H',
                    f"Cells (n={cells_mask.sum():,})",
                    gate_coords=st.session_state.gates['singlets'],
                    parent_mask=st.session_state.masks['cells']
                )
                st.plotly_chart(fig2, use_container_width=True, key="plot_singlets")
                
                singlets_mask = apply_gate(data, fsc_a, fsc_h, st.session_state.gates['singlets'], 
                                          st.session_state.masks['cells'])
                st.session_state.masks['singlets'] = singlets_mask
                
                n_parent = cells_mask.sum()
                st.markdown(f"**Singlets:** {singlets_mask.sum():,} ({singlets_mask.sum()/n_parent*100:.1f}%)")
                all_stats.append({'Population': 'Single Cells', 'Parent': 'Cells',
                                 'Count': singlets_mask.sum(), '% Parent': round(singlets_mask.sum()/n_parent*100, 1)})
        
        # Plot 3: Live
        with col3:
            st.markdown("##### Live (Live/Dead vs SSC-A)")
            if livedead and ssc_a and 'singlets' in st.session_state.masks:
                fig3, n_live, pct_live = create_interactive_plot(
                    data, livedead, ssc_a, 'Live/Dead', 'SSC-A',
                    f"Singlets (n={singlets_mask.sum():,})",
                    gate_coords=st.session_state.gates['live'],
                    parent_mask=st.session_state.masks['singlets']
                )
                st.plotly_chart(fig3, use_container_width=True, key="plot_live")
                
                live_mask = apply_gate(data, livedead, ssc_a, st.session_state.gates['live'],
                                      st.session_state.masks['singlets'])
                st.session_state.masks['live'] = live_mask
                
                n_parent = singlets_mask.sum()
                st.markdown(f"**Live:** {live_mask.sum():,} ({live_mask.sum()/n_parent*100:.1f}%)")
                all_stats.append({'Population': 'Live', 'Parent': 'Single Cells',
                                 'Count': live_mask.sum(), '% Parent': round(live_mask.sum()/n_parent*100, 1)})
        
        # Plot 4: hCD45+
        with col4:
            st.markdown("##### hCD45+ (hCD45 vs SSC-A)")
            if hcd45 and ssc_a and 'live' in st.session_state.masks:
                fig4, n_hcd45, pct_hcd45 = create_interactive_plot(
                    data, hcd45, ssc_a, 'hCD45', 'SSC-A',
                    f"Live (n={live_mask.sum():,})",
                    gate_coords=st.session_state.gates['hcd45'],
                    parent_mask=st.session_state.masks['live']
                )
                st.plotly_chart(fig4, use_container_width=True, key="plot_hcd45")
                
                hcd45_mask = apply_gate(data, hcd45, ssc_a, st.session_state.gates['hcd45'],
                                       st.session_state.masks['live'])
                st.session_state.masks['hcd45'] = hcd45_mask
                
                n_parent = live_mask.sum()
                st.markdown(f"**hCD45+:** {hcd45_mask.sum():,} ({hcd45_mask.sum()/n_parent*100:.1f}%)")
                all_stats.append({'Population': 'hCD45+ (Leucocytes)', 'Parent': 'Live',
                                 'Count': hcd45_mask.sum(), '% Parent': round(hcd45_mask.sum()/n_parent*100, 1)})
        
        # ===== ROW 2: SOUS-POPULATIONS =====
        st.markdown("### 2️⃣ Sous-Populations (Quadrants)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Plot 5: NK cells
        with col1:
            st.markdown("##### NK cells (CD56 vs CD16)")
            if cd56 and cd16 and 'hcd45' in st.session_state.masks:
                fig5, _, _ = create_interactive_plot(
                    data, cd56, cd16, 'CD56', 'CD16',
                    f"hCD45+ (n={hcd45_mask.sum():,})",
                    gate_coords=st.session_state.gates['nk'],
                    parent_mask=st.session_state.masks['hcd45'],
                    is_quadrant=True
                )
                st.plotly_chart(fig5, use_container_width=True, key="plot_nk")
        
        # Plot 6: T/B cells
        with col2:
            st.markdown("##### T/B cells (CD3 vs CD19)")
            if cd3 and cd19 and 'hcd45' in st.session_state.masks:
                fig6, _, _ = create_interactive_plot(
                    data, cd3, cd19, 'CD3', 'CD19',
                    f"hCD45+ (n={hcd45_mask.sum():,})",
                    gate_coords=st.session_state.gates['tb'],
                    parent_mask=st.session_state.masks['hcd45'],
                    is_quadrant=True
                )
                st.plotly_chart(fig6, use_container_width=True, key="plot_tb")
                
                # Créer mask T cells
                t_thresh_x = st.session_state.gates['tb']['x_thresh']
                t_thresh_y = st.session_state.gates['tb']['y_thresh']
                t_mask = st.session_state.masks['hcd45'] & (data[cd3] >= t_thresh_x) & (data[cd19] < t_thresh_y)
                st.session_state.masks['t_cells'] = t_mask
                
                b_mask = st.session_state.masks['hcd45'] & (data[cd3] < t_thresh_x) & (data[cd19] >= t_thresh_y)
                
                n_parent = hcd45_mask.sum()
                all_stats.append({'Population': 'T cells', 'Parent': 'hCD45+',
                                 'Count': t_mask.sum(), '% Parent': round(t_mask.sum()/n_parent*100, 1)})
                all_stats.append({'Population': 'B cells', 'Parent': 'hCD45+',
                                 'Count': b_mask.sum(), '% Parent': round(b_mask.sum()/n_parent*100, 1)})
        
        # Plot 7: CD4/CD8
        with col3:
            st.markdown("##### CD4/CD8 (CD4 vs CD8)")
            if cd4 and cd8 and 't_cells' in st.session_state.masks:
                t_cells_mask = st.session_state.masks['t_cells']
                fig7, _, _ = create_interactive_plot(
                    data, cd4, cd8, 'CD4', 'CD8',
                    f"T cells (n={t_cells_mask.sum():,})",
                    gate_coords=st.session_state.gates['cd4cd8'],
                    parent_mask=t_cells_mask,
                    is_quadrant=True
                )
                st.plotly_chart(fig7, use_container_width=True, key="plot_cd4cd8")
                
                # Créer mask CD4+
                cd4_thresh_x = st.session_state.gates['cd4cd8']['x_thresh']
                cd4_thresh_y = st.session_state.gates['cd4cd8']['y_thresh']
                cd4_mask = t_cells_mask & (data[cd4] >= cd4_thresh_x) & (data[cd8] < cd4_thresh_y)
                st.session_state.masks['cd4_cells'] = cd4_mask
                
                cd8_mask = t_cells_mask & (data[cd4] < cd4_thresh_x) & (data[cd8] >= cd4_thresh_y)
                
                n_parent = t_cells_mask.sum()
                if n_parent > 0:
                    all_stats.append({'Population': 'CD4+ T cells', 'Parent': 'T cells',
                                     'Count': cd4_mask.sum(), '% Parent': round(cd4_mask.sum()/n_parent*100, 1)})
                    all_stats.append({'Population': 'CD8+ T cells', 'Parent': 'T cells',
                                     'Count': cd8_mask.sum(), '% Parent': round(cd8_mask.sum()/n_parent*100, 1)})
        
        # Plot 8: Treg
        with col4:
            st.markdown("##### Treg (FoxP3 vs CD25)")
            if foxp3 and cd25 and 'cd4_cells' in st.session_state.masks:
                cd4_cells_mask = st.session_state.masks['cd4_cells']
                fig8, _, _ = create_interactive_plot(
                    data, foxp3, cd25, 'FoxP3', 'CD25',
                    f"CD4+ (n={cd4_cells_mask.sum():,})",
                    gate_coords=st.session_state.gates['treg'],
                    parent_mask=cd4_cells_mask,
                    is_quadrant=True
                )
                st.plotly_chart(fig8, use_container_width=True, key="plot_treg")
                
                # Treg stats
                treg_thresh_x = st.session_state.gates['treg']['x_thresh']
                treg_thresh_y = st.session_state.gates['treg']['y_thresh']
                treg_mask = cd4_cells_mask & (data[foxp3] >= treg_thresh_x) & (data[cd25] >= treg_thresh_y)
                
                n_parent = cd4_cells_mask.sum()
                if n_parent > 0:
                    all_stats.append({'Population': 'Treg (FoxP3+CD25+)', 'Parent': 'CD4+ T cells',
                                     'Count': treg_mask.sum(), '% Parent': round(treg_mask.sum()/n_parent*100, 1)})
        
        # ==================== STATISTIQUES ====================
        
        st.markdown("---")
        st.markdown("### 📊 Résumé des Populations")
        
        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_df['% Total'] = (stats_df['Count'] / n_total * 100).round(2)
            
            st.dataframe(stats_df, use_container_width=True)
            
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
                # Export Excel
                try:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        stats_df.to_excel(writer, sheet_name='Populations', index=False)
                    output.seek(0)
                    st.download_button(
                        "📥 Télécharger Excel",
                        output,
                        f"{reader.filename}_populations.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except:
                    pass
        
        # ==================== AJUSTEMENT MANUEL ====================
        
        with st.expander("⚙️ Ajustement manuel des seuils (optionnel)"):
            st.markdown("*Utilisez ces sliders si vous préférez un ajustement précis*")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**Gate Cells**")
                if fsc_a:
                    new_x0 = st.number_input("FSC-A min", value=float(st.session_state.gates['cells']['x0']), key="cells_x0")
                    new_x1 = st.number_input("FSC-A max", value=float(st.session_state.gates['cells']['x1']), key="cells_x1")
                    st.session_state.gates['cells']['x0'] = new_x0
                    st.session_state.gates['cells']['x1'] = new_x1
            
            with col2:
                st.markdown("**Quadrant T/B**")
                if cd3:
                    new_cd3 = st.number_input("Seuil CD3", value=float(st.session_state.gates['tb']['x_thresh']), key="tb_x")
                    new_cd19 = st.number_input("Seuil CD19", value=float(st.session_state.gates['tb']['y_thresh']), key="tb_y")
                    st.session_state.gates['tb']['x_thresh'] = new_cd3
                    st.session_state.gates['tb']['y_thresh'] = new_cd19
            
            with col3:
                st.markdown("**Quadrant CD4/CD8**")
                if cd4:
                    new_cd4 = st.number_input("Seuil CD4", value=float(st.session_state.gates['cd4cd8']['x_thresh']), key="cd4_x")
                    new_cd8 = st.number_input("Seuil CD8", value=float(st.session_state.gates['cd4cd8']['y_thresh']), key="cd8_y")
                    st.session_state.gates['cd4cd8']['x_thresh'] = new_cd4
                    st.session_state.gates['cd4cd8']['y_thresh'] = new_cd8
            
            with col4:
                st.markdown("**Quadrant Treg**")
                if foxp3:
                    new_foxp3 = st.number_input("Seuil FoxP3", value=float(st.session_state.gates['treg']['x_thresh']), key="treg_x")
                    new_cd25 = st.number_input("Seuil CD25", value=float(st.session_state.gates['treg']['y_thresh']), key="treg_y")
                    st.session_state.gates['treg']['x_thresh'] = new_foxp3
                    st.session_state.gates['treg']['y_thresh'] = new_cd25
            
            if st.button("🔄 Recalculer avec nouveaux seuils", type="primary"):
                st.rerun()
    
    except Exception as e:
        st.error(f"Erreur: {e}")
        st.exception(e)

else:
    st.info("👆 Uploadez un fichier FCS pour commencer")
    
    st.markdown("""
    ### 📌 Comment utiliser cette application
    
    1. **Uploadez** votre fichier FCS
    2. **Ajustez les gates** directement sur les graphiques :
       - **Rectangles** : cliquez et glissez les bords/coins
       - **Lignes de quadrant** : cliquez et glissez pour déplacer
    3. **Visualisez** les statistiques mises à jour automatiquement
    4. **Exportez** vos résultats en CSV ou Excel
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🔬 <b>FACS Analysis - Gates Interactifs</b><br>
    Ajustez les gates directement sur les graphiques
</div>
""", unsafe_allow_html=True)
