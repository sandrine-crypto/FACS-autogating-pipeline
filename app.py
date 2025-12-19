#!/usr/bin/env python3
"""
FACS Autogating - Style FlowJo avec Gates Ajustables
- Gates visibles sur les graphiques (rectangles)
- Sliders pour ajuster chaque gate en temps réel
- Workflow immunophénotypage complet
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from pathlib import Path
import tempfile
import io
from datetime import datetime
import flowio
import re

# Configuration
st.set_page_config(
    page_title="FACS - Gates Ajustables",
    page_icon="🔬",
    layout="wide"
)

# CSS
st.markdown("""
    <style>
    .main-header { font-size: 2rem; color: #2c3e50; text-align: center; }
    .gate-box { background: #e8f4f8; padding: 0.8rem; border-radius: 0.5rem; 
                border-left: 4px solid #3498db; margin: 0.5rem 0; }
    .stSlider > div > div { padding-top: 0.5rem; }
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
            
            # Extraire marqueur
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
    """Trouve un canal par mots-clés"""
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


def create_gated_plot(ax, x_data, y_data, x_label, y_label, title,
                      gate_x_min=None, gate_x_max=None, 
                      gate_y_min=None, gate_y_max=None,
                      gate_name="", show_gate=True, cmap='jet'):
    """Crée un plot avec gate rectangulaire visible et ajustable"""
    
    # Filtrer données valides
    valid = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
    x_plot = biex_transform(x_data[valid])
    y_plot = biex_transform(y_data[valid])
    
    if len(x_plot) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return 0, 0
    
    # Plot pseudocolor
    try:
        ax.hist2d(x_plot, y_plot, bins=80, cmap=cmap, norm=LogNorm(), cmin=1, rasterized=True)
    except:
        ax.scatter(x_plot, y_plot, s=1, c='blue', alpha=0.3, rasterized=True)
    
    # Calculer les limites du plot
    x_min_plot, x_max_plot = np.percentile(x_plot, [0.5, 99.5])
    y_min_plot, y_max_plot = np.percentile(y_plot, [0.5, 99.5])
    ax.set_xlim(x_min_plot - 5, x_max_plot + 5)
    ax.set_ylim(y_min_plot - 5, y_max_plot + 5)
    
    # Dessiner le gate rectangulaire
    n_in_gate = 0
    pct = 0
    
    if show_gate and gate_x_min is not None:
        # Transformer les seuils
        gx_min_t = biex_transform([gate_x_min])[0]
        gx_max_t = biex_transform([gate_x_max])[0]
        gy_min_t = biex_transform([gate_y_min])[0]
        gy_max_t = biex_transform([gate_y_max])[0]
        
        # Dessiner le rectangle
        rect = patches.Rectangle(
            (gx_min_t, gy_min_t), 
            gx_max_t - gx_min_t, 
            gy_max_t - gy_min_t,
            linewidth=2, edgecolor='red', facecolor='none', linestyle='-'
        )
        ax.add_patch(rect)
        
        # Calculer % dans le gate
        in_gate = (x_data >= gate_x_min) & (x_data <= gate_x_max) & \
                  (y_data >= gate_y_min) & (y_data <= gate_y_max) & valid
        n_in_gate = in_gate.sum()
        pct = n_in_gate / valid.sum() * 100 if valid.sum() > 0 else 0
        
        # Afficher le nom et % du gate
        ax.text(gx_max_t + 2, gy_max_t, f'{gate_name}\n{pct:.1f}%\n({n_in_gate:,})', 
               fontsize=9, fontweight='bold', color='red',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return n_in_gate, pct


def create_quadrant_plot(ax, x_data, y_data, x_label, y_label, title,
                         x_threshold, y_threshold, quadrant_names, cmap='jet'):
    """Crée un plot avec quadrants ajustables"""
    
    valid = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
    x_plot = biex_transform(x_data[valid])
    y_plot = biex_transform(y_data[valid])
    
    if len(x_plot) == 0:
        return {}
    
    # Plot
    try:
        ax.hist2d(x_plot, y_plot, bins=80, cmap=cmap, norm=LogNorm(), cmin=1, rasterized=True)
    except:
        ax.scatter(x_plot, y_plot, s=1, c='blue', alpha=0.3, rasterized=True)
    
    # Limites
    x_min_plot, x_max_plot = np.percentile(x_plot, [0.5, 99.5])
    y_min_plot, y_max_plot = np.percentile(y_plot, [0.5, 99.5])
    ax.set_xlim(x_min_plot - 5, x_max_plot + 5)
    ax.set_ylim(y_min_plot - 5, y_max_plot + 5)
    
    # Transformer les seuils
    x_thresh_t = biex_transform([x_threshold])[0]
    y_thresh_t = biex_transform([y_threshold])[0]
    
    # Dessiner les lignes de quadrant
    ax.axvline(x=x_thresh_t, color='black', linewidth=2, linestyle='-')
    ax.axhline(y=y_thresh_t, color='black', linewidth=2, linestyle='-')
    
    # Calculer % dans chaque quadrant
    n_total = valid.sum()
    quadrant_stats = {}
    
    # Q1: haut-droite (++), Q2: haut-gauche (-+), Q3: bas-gauche (--), Q4: bas-droite (+-)
    conditions = [
        (x_data >= x_threshold) & (y_data >= y_threshold),  # Q1: ++
        (x_data < x_threshold) & (y_data >= y_threshold),   # Q2: -+
        (x_data < x_threshold) & (y_data < y_threshold),    # Q3: --
        (x_data >= x_threshold) & (y_data < y_threshold),   # Q4: +-
    ]
    
    positions = [
        (0.95, 0.95, 'right', 'top'),
        (0.05, 0.95, 'left', 'top'),
        (0.05, 0.05, 'left', 'bottom'),
        (0.95, 0.05, 'right', 'bottom'),
    ]
    
    for i, (cond, name) in enumerate(zip(conditions, quadrant_names)):
        n_quad = (cond & valid).sum()
        pct = n_quad / n_total * 100 if n_total > 0 else 0
        quadrant_stats[name] = {'count': n_quad, 'pct': pct}
        
        px, py, ha, va = positions[i]
        ax.text(px, py, f'{name}\n{pct:.1f}%', transform=ax.transAxes,
               fontsize=8, fontweight='bold', ha=ha, va=va,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    return quadrant_stats


def create_histogram_with_gate(ax, data, channel, marker, threshold, gate_name):
    """Crée un histogramme avec gate ajustable"""
    
    valid = np.isfinite(data) & (data > 0)
    plot_data = biex_transform(data[valid])
    
    if len(plot_data) == 0:
        return 0, 0
    
    ax.hist(plot_data, bins=80, color='lightgray', edgecolor='darkgray', linewidth=0.5)
    
    # Gate
    thresh_t = biex_transform([threshold])[0]
    ax.axvline(x=thresh_t, color='red', linewidth=2, linestyle='-')
    
    # % positif
    n_pos = (data[valid] > threshold).sum()
    pct = n_pos / valid.sum() * 100 if valid.sum() > 0 else 0
    
    ax.text(0.95, 0.95, f'{gate_name}\n{pct:.1f}%', transform=ax.transAxes,
           fontsize=9, fontweight='bold', ha='right', va='top', color='red',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_xlabel(f'{marker}', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(marker, fontsize=11, fontweight='bold')
    
    return n_pos, pct


# ==================== INTERFACE STREAMLIT ====================

st.markdown('<h1 class="main-header">🔬 FACS - Gates Ajustables (Style FlowJo)</h1>', unsafe_allow_html=True)

# Session state pour les gates
if 'gates_params' not in st.session_state:
    st.session_state.gates_params = {}
if 'reader' not in st.session_state:
    st.session_state.reader = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Upload
uploaded_file = st.file_uploader("📁 Fichier FCS", type=['fcs'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    try:
        # Charger les données
        if not st.session_state.data_loaded or st.session_state.get('current_file') != uploaded_file.name:
            with st.spinner("Chargement..."):
                reader = FCSReader(tmp_path)
                st.session_state.reader = reader
                st.session_state.data_loaded = True
                st.session_state.current_file = uploaded_file.name
                
                # Initialiser les paramètres de gates avec valeurs par défaut
                data = reader.data
                
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
                
                # Stocker les canaux trouvés
                st.session_state.channels = {
                    'FSC-A': fsc_a, 'FSC-H': fsc_h, 'SSC-A': ssc_a,
                    'LiveDead': livedead, 'hCD45': hcd45, 'mCD45': mcd45,
                    'CD3': cd3, 'CD19': cd19, 'CD4': cd4, 'CD8': cd8,
                    'CD56': cd56, 'CD16': cd16, 'FoxP3': foxp3, 'CD25': cd25
                }
                
                # Calculer les seuils par défaut
                if fsc_a and ssc_a:
                    st.session_state.gates_params['cells'] = {
                        'x_min': float(np.percentile(data[fsc_a], 5)),
                        'x_max': float(np.percentile(data[fsc_a], 99)),
                        'y_min': float(np.percentile(data[ssc_a], 5)),
                        'y_max': float(np.percentile(data[ssc_a], 99)),
                    }
                
                if fsc_a and fsc_h:
                    st.session_state.gates_params['singlets'] = {
                        'x_min': float(np.percentile(data[fsc_a], 2)),
                        'x_max': float(np.percentile(data[fsc_a], 98)),
                        'y_min': float(np.percentile(data[fsc_h], 2)),
                        'y_max': float(np.percentile(data[fsc_h], 98)),
                    }
                
                if livedead:
                    st.session_state.gates_params['live'] = {
                        'threshold': float(np.percentile(data[livedead], 85))
                    }
                
                if hcd45:
                    st.session_state.gates_params['hcd45'] = {
                        'threshold': float(np.percentile(data[hcd45], 15))
                    }
                
                if cd3 and cd19:
                    st.session_state.gates_params['cd3_cd19'] = {
                        'x_threshold': float(np.percentile(data[cd3], 40)),
                        'y_threshold': float(np.percentile(data[cd19], 75)),
                    }
                
                if cd4 and cd8:
                    st.session_state.gates_params['cd4_cd8'] = {
                        'x_threshold': float(np.percentile(data[cd4], 35)),
                        'y_threshold': float(np.percentile(data[cd8], 75)),
                    }
                
                if cd56 and cd16:
                    st.session_state.gates_params['nk'] = {
                        'x_threshold': float(np.percentile(data[cd56], 65)),
                        'y_threshold': float(np.percentile(data[cd16], 65)),
                    }
                
                if foxp3 and cd25:
                    st.session_state.gates_params['treg'] = {
                        'x_threshold': float(np.percentile(data[foxp3], 90)),
                        'y_threshold': float(np.percentile(data[cd25], 85)),
                    }
        
        reader = st.session_state.reader
        data = reader.data
        channels = st.session_state.channels
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        col1.metric("Événements", f"{len(data):,}")
        col2.metric("Canaux", len(reader.channels))
        col3.metric("Fichier", reader.filename[:25])
        
        st.markdown("---")
        
        # ==================== CONTRÔLES DES GATES ====================
        
        with st.expander("⚙️ **AJUSTEMENT DES GATES** (cliquez pour ouvrir)", expanded=True):
            
            st.markdown("### 🎯 Ajustez les seuils des gates")
            
            tab_g1, tab_g2, tab_g3, tab_g4 = st.tabs([
                "1️⃣ Cells/Singlets", 
                "2️⃣ Live/hCD45", 
                "3️⃣ T/B/NK cells",
                "4️⃣ CD4/CD8/Treg"
            ])
            
            # TAB 1: Cells & Singlets
            with tab_g1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔹 Gate Cells (FSC-A vs SSC-A)")
                    if 'cells' in st.session_state.gates_params:
                        fsc_a = channels['FSC-A']
                        ssc_a = channels['SSC-A']
                        fsc_min = float(data[fsc_a].min())
                        fsc_max = float(data[fsc_a].max())
                        ssc_min = float(data[ssc_a].min())
                        ssc_max = float(data[ssc_a].max())
                        
                        st.session_state.gates_params['cells']['x_min'] = st.slider(
                            "FSC-A min", fsc_min, fsc_max, 
                            st.session_state.gates_params['cells']['x_min'],
                            key='cells_xmin'
                        )
                        st.session_state.gates_params['cells']['x_max'] = st.slider(
                            "FSC-A max", fsc_min, fsc_max,
                            st.session_state.gates_params['cells']['x_max'],
                            key='cells_xmax'
                        )
                        st.session_state.gates_params['cells']['y_min'] = st.slider(
                            "SSC-A min", ssc_min, ssc_max,
                            st.session_state.gates_params['cells']['y_min'],
                            key='cells_ymin'
                        )
                        st.session_state.gates_params['cells']['y_max'] = st.slider(
                            "SSC-A max", ssc_min, ssc_max,
                            st.session_state.gates_params['cells']['y_max'],
                            key='cells_ymax'
                        )
                
                with col2:
                    st.markdown("#### 🔹 Gate Singlets (FSC-A vs FSC-H)")
                    if 'singlets' in st.session_state.gates_params:
                        fsc_h = channels['FSC-H']
                        fsch_min = float(data[fsc_h].min())
                        fsch_max = float(data[fsc_h].max())
                        
                        st.session_state.gates_params['singlets']['x_min'] = st.slider(
                            "FSC-A min (singlets)", fsc_min, fsc_max,
                            st.session_state.gates_params['singlets']['x_min'],
                            key='sing_xmin'
                        )
                        st.session_state.gates_params['singlets']['x_max'] = st.slider(
                            "FSC-A max (singlets)", fsc_min, fsc_max,
                            st.session_state.gates_params['singlets']['x_max'],
                            key='sing_xmax'
                        )
                        st.session_state.gates_params['singlets']['y_min'] = st.slider(
                            "FSC-H min", fsch_min, fsch_max,
                            st.session_state.gates_params['singlets']['y_min'],
                            key='sing_ymin'
                        )
                        st.session_state.gates_params['singlets']['y_max'] = st.slider(
                            "FSC-H max", fsch_min, fsch_max,
                            st.session_state.gates_params['singlets']['y_max'],
                            key='sing_ymax'
                        )
            
            # TAB 2: Live & hCD45
            with tab_g2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔹 Gate Live (Live/Dead)")
                    if 'live' in st.session_state.gates_params and channels['LiveDead']:
                        ld = channels['LiveDead']
                        ld_min = float(data[ld].min())
                        ld_max = float(data[ld].max())
                        
                        st.session_state.gates_params['live']['threshold'] = st.slider(
                            "Seuil Live/Dead (en dessous = vivant)", ld_min, ld_max,
                            st.session_state.gates_params['live']['threshold'],
                            key='live_thresh'
                        )
                        st.info("Les cellules **sous** ce seuil sont considérées vivantes")
                
                with col2:
                    st.markdown("#### 🔹 Gate hCD45+")
                    if 'hcd45' in st.session_state.gates_params and channels['hCD45']:
                        hcd45 = channels['hCD45']
                        hcd45_min = float(data[hcd45].min())
                        hcd45_max = float(data[hcd45].max())
                        
                        st.session_state.gates_params['hcd45']['threshold'] = st.slider(
                            "Seuil hCD45 (au dessus = positif)", hcd45_min, hcd45_max,
                            st.session_state.gates_params['hcd45']['threshold'],
                            key='hcd45_thresh'
                        )
                        st.info("Les cellules **au-dessus** de ce seuil sont hCD45+")
            
            # TAB 3: T/B/NK cells
            with tab_g3:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔹 Quadrant CD3 vs CD19 (T/B cells)")
                    if 'cd3_cd19' in st.session_state.gates_params:
                        cd3 = channels['CD3']
                        cd19 = channels['CD19']
                        
                        if cd3 and cd19:
                            cd3_min = float(data[cd3].min())
                            cd3_max = float(data[cd3].max())
                            cd19_min = float(data[cd19].min())
                            cd19_max = float(data[cd19].max())
                            
                            st.session_state.gates_params['cd3_cd19']['x_threshold'] = st.slider(
                                "Seuil CD3", cd3_min, cd3_max,
                                st.session_state.gates_params['cd3_cd19']['x_threshold'],
                                key='cd3_thresh'
                            )
                            st.session_state.gates_params['cd3_cd19']['y_threshold'] = st.slider(
                                "Seuil CD19", cd19_min, cd19_max,
                                st.session_state.gates_params['cd3_cd19']['y_threshold'],
                                key='cd19_thresh'
                            )
                
                with col2:
                    st.markdown("#### 🔹 Quadrant CD56 vs CD16 (NK cells)")
                    if 'nk' in st.session_state.gates_params:
                        cd56 = channels['CD56']
                        cd16 = channels['CD16']
                        
                        if cd56 and cd16:
                            cd56_min = float(data[cd56].min())
                            cd56_max = float(data[cd56].max())
                            cd16_min = float(data[cd16].min())
                            cd16_max = float(data[cd16].max())
                            
                            st.session_state.gates_params['nk']['x_threshold'] = st.slider(
                                "Seuil CD56", cd56_min, cd56_max,
                                st.session_state.gates_params['nk']['x_threshold'],
                                key='cd56_thresh'
                            )
                            st.session_state.gates_params['nk']['y_threshold'] = st.slider(
                                "Seuil CD16", cd16_min, cd16_max,
                                st.session_state.gates_params['nk']['y_threshold'],
                                key='cd16_thresh'
                            )
            
            # TAB 4: CD4/CD8/Treg
            with tab_g4:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔹 Quadrant CD4 vs CD8")
                    if 'cd4_cd8' in st.session_state.gates_params:
                        cd4 = channels['CD4']
                        cd8 = channels['CD8']
                        
                        if cd4 and cd8:
                            cd4_min = float(data[cd4].min())
                            cd4_max = float(data[cd4].max())
                            cd8_min = float(data[cd8].min())
                            cd8_max = float(data[cd8].max())
                            
                            st.session_state.gates_params['cd4_cd8']['x_threshold'] = st.slider(
                                "Seuil CD4", cd4_min, cd4_max,
                                st.session_state.gates_params['cd4_cd8']['x_threshold'],
                                key='cd4_thresh'
                            )
                            st.session_state.gates_params['cd4_cd8']['y_threshold'] = st.slider(
                                "Seuil CD8", cd8_min, cd8_max,
                                st.session_state.gates_params['cd4_cd8']['y_threshold'],
                                key='cd8_thresh'
                            )
                
                with col2:
                    st.markdown("#### 🔹 Quadrant FoxP3 vs CD25 (Treg)")
                    if 'treg' in st.session_state.gates_params:
                        foxp3 = channels['FoxP3']
                        cd25 = channels['CD25']
                        
                        if foxp3 and cd25:
                            foxp3_min = float(data[foxp3].min())
                            foxp3_max = float(data[foxp3].max())
                            cd25_min = float(data[cd25].min())
                            cd25_max = float(data[cd25].max())
                            
                            st.session_state.gates_params['treg']['x_threshold'] = st.slider(
                                "Seuil FoxP3", foxp3_min, foxp3_max,
                                st.session_state.gates_params['treg']['x_threshold'],
                                key='foxp3_thresh'
                            )
                            st.session_state.gates_params['treg']['y_threshold'] = st.slider(
                                "Seuil CD25", cd25_min, cd25_max,
                                st.session_state.gates_params['treg']['y_threshold'],
                                key='cd25_thresh'
                            )
        
        st.markdown("---")
        
        # ==================== VISUALISATION ====================
        
        st.markdown("### 📊 Visualisation avec Gates")
        
        if st.button("🔄 **Générer/Actualiser les Graphiques**", type="primary", use_container_width=True):
            
            with st.spinner("Génération des graphiques avec les gates ajustés..."):
                
                # Créer la figure
                fig = plt.figure(figsize=(20, 12), dpi=100)
                
                all_stats = []
                n_total = len(data)
                
                # ===== ROW 1: GATING HIÉRARCHIQUE =====
                
                # Plot 1: Cells (FSC-A vs SSC-A)
                ax1 = fig.add_subplot(2, 4, 1)
                fsc_a = channels['FSC-A']
                ssc_a = channels['SSC-A']
                
                if fsc_a and ssc_a and 'cells' in st.session_state.gates_params:
                    gp = st.session_state.gates_params['cells']
                    n_cells, pct_cells = create_gated_plot(
                        ax1, data[fsc_a].values, data[ssc_a].values,
                        'FSC-A', 'SSC-A', f'{reader.filename}\nUngated: {n_total:,}',
                        gp['x_min'], gp['x_max'], gp['y_min'], gp['y_max'],
                        gate_name='Cells'
                    )
                    
                    # Créer le mask pour Cells
                    cells_mask = (data[fsc_a] >= gp['x_min']) & (data[fsc_a] <= gp['x_max']) & \
                                 (data[ssc_a] >= gp['y_min']) & (data[ssc_a] <= gp['y_max'])
                    
                    all_stats.append({'Population': 'Cells', 'Parent': 'Ungated', 
                                     'Count': n_cells, '% Parent': pct_cells})
                else:
                    cells_mask = pd.Series(True, index=data.index)
                    n_cells = len(data)
                
                # Plot 2: Singlets (FSC-A vs FSC-H)
                ax2 = fig.add_subplot(2, 4, 2)
                fsc_h = channels['FSC-H']
                
                if fsc_a and fsc_h and 'singlets' in st.session_state.gates_params:
                    cells_data = data[cells_mask]
                    gp = st.session_state.gates_params['singlets']
                    
                    n_singlets, pct_singlets = create_gated_plot(
                        ax2, cells_data[fsc_a].values, cells_data[fsc_h].values,
                        'FSC-A', 'FSC-H', f'Cells: {n_cells:,}',
                        gp['x_min'], gp['x_max'], gp['y_min'], gp['y_max'],
                        gate_name='Single Cells'
                    )
                    
                    singlets_mask = cells_mask & \
                                   (data[fsc_a] >= gp['x_min']) & (data[fsc_a] <= gp['x_max']) & \
                                   (data[fsc_h] >= gp['y_min']) & (data[fsc_h] <= gp['y_max'])
                    
                    all_stats.append({'Population': 'Single Cells', 'Parent': 'Cells',
                                     'Count': singlets_mask.sum(), '% Parent': round(singlets_mask.sum()/n_cells*100, 1)})
                else:
                    singlets_mask = cells_mask
                
                # Plot 3: Live
                ax3 = fig.add_subplot(2, 4, 3)
                livedead = channels['LiveDead']
                
                if livedead and ssc_a and 'live' in st.session_state.gates_params:
                    singlets_data = data[singlets_mask]
                    thresh = st.session_state.gates_params['live']['threshold']
                    
                    # Plot avec ligne de gate horizontale
                    valid = singlets_mask & np.isfinite(data[livedead]) & (data[livedead] > 0)
                    x_plot = biex_transform(data.loc[valid, livedead].values)
                    y_plot = biex_transform(data.loc[valid, ssc_a].values)
                    
                    try:
                        ax3.hist2d(x_plot, y_plot, bins=80, cmap='jet', norm=LogNorm(), cmin=1)
                    except:
                        ax3.scatter(x_plot, y_plot, s=1, alpha=0.3)
                    
                    thresh_t = biex_transform([thresh])[0]
                    ax3.axvline(x=thresh_t, color='red', linewidth=2)
                    
                    live_mask = singlets_mask & (data[livedead] < thresh)
                    n_live = live_mask.sum()
                    n_sing = singlets_mask.sum()
                    pct_live = n_live / n_sing * 100 if n_sing > 0 else 0
                    
                    ax3.text(0.05, 0.95, f'Live\n{pct_live:.1f}%', transform=ax3.transAxes,
                            fontsize=9, fontweight='bold', va='top', color='red',
                            bbox=dict(facecolor='white', alpha=0.8))
                    ax3.set_xlabel('Live/Dead', fontsize=10, fontweight='bold')
                    ax3.set_ylabel('SSC-A', fontsize=10, fontweight='bold')
                    ax3.set_title(f'Single Cells: {n_sing:,}', fontsize=11, fontweight='bold')
                    
                    all_stats.append({'Population': 'Live', 'Parent': 'Single Cells',
                                     'Count': n_live, '% Parent': round(pct_live, 1)})
                else:
                    live_mask = singlets_mask
                    n_live = live_mask.sum()
                
                # Plot 4: hCD45+
                ax4 = fig.add_subplot(2, 4, 4)
                hcd45 = channels['hCD45']
                
                if hcd45 and 'hcd45' in st.session_state.gates_params:
                    live_data = data[live_mask]
                    thresh = st.session_state.gates_params['hcd45']['threshold']
                    
                    valid = live_mask & np.isfinite(data[hcd45]) & (data[hcd45] > 0)
                    x_plot = biex_transform(data.loc[valid, hcd45].values)
                    y_plot = biex_transform(data.loc[valid, ssc_a].values) if ssc_a else x_plot
                    
                    try:
                        ax4.hist2d(x_plot, y_plot, bins=80, cmap='jet', norm=LogNorm(), cmin=1)
                    except:
                        ax4.scatter(x_plot, y_plot, s=1, alpha=0.3)
                    
                    thresh_t = biex_transform([thresh])[0]
                    ax4.axvline(x=thresh_t, color='red', linewidth=2)
                    
                    hcd45_mask = live_mask & (data[hcd45] > thresh)
                    n_hcd45 = hcd45_mask.sum()
                    pct_hcd45 = n_hcd45 / n_live * 100 if n_live > 0 else 0
                    
                    ax4.text(0.95, 0.95, f'hCD45+\n{pct_hcd45:.1f}%', transform=ax4.transAxes,
                            fontsize=9, fontweight='bold', va='top', ha='right', color='red',
                            bbox=dict(facecolor='white', alpha=0.8))
                    ax4.set_xlabel('hCD45 (PerCP)', fontsize=10, fontweight='bold')
                    ax4.set_ylabel('SSC-A', fontsize=10, fontweight='bold')
                    ax4.set_title(f'Live: {n_live:,}', fontsize=11, fontweight='bold')
                    
                    all_stats.append({'Population': 'hCD45+ (Leucocytes)', 'Parent': 'Live',
                                     'Count': n_hcd45, '% Parent': round(pct_hcd45, 1)})
                else:
                    hcd45_mask = live_mask
                    n_hcd45 = hcd45_mask.sum()
                
                # ===== ROW 2: SOUS-POPULATIONS =====
                
                # Plot 5: NK cells (CD56 vs CD16)
                ax5 = fig.add_subplot(2, 4, 5)
                cd56 = channels['CD56']
                cd16 = channels['CD16']
                
                if cd56 and cd16 and 'nk' in st.session_state.gates_params:
                    leuco_data = data[hcd45_mask]
                    gp = st.session_state.gates_params['nk']
                    
                    quad_stats = create_quadrant_plot(
                        ax5, leuco_data[cd56].values, leuco_data[cd16].values,
                        'CD56', 'CD16', f'hCD45+: {n_hcd45:,}',
                        gp['x_threshold'], gp['y_threshold'],
                        ['NK cells', 'CD16-CD56+', 'DN', 'CD16+CD56-']
                    )
                    
                    for name, stats in quad_stats.items():
                        all_stats.append({'Population': f'NK-{name}', 'Parent': 'hCD45+',
                                         'Count': stats['count'], '% Parent': round(stats['pct'], 1)})
                
                # Plot 6: T/B cells (CD3 vs CD19)
                ax6 = fig.add_subplot(2, 4, 6)
                cd3 = channels['CD3']
                cd19 = channels['CD19']
                
                if cd3 and cd19 and 'cd3_cd19' in st.session_state.gates_params:
                    leuco_data = data[hcd45_mask]
                    gp = st.session_state.gates_params['cd3_cd19']
                    
                    quad_stats = create_quadrant_plot(
                        ax6, leuco_data[cd3].values, leuco_data[cd19].values,
                        'CD3', 'CD19', f'hCD45+: {n_hcd45:,}',
                        gp['x_threshold'], gp['y_threshold'],
                        ['B cells', 'DP', 'DN', 'T cells']
                    )
                    
                    # Créer le mask T cells
                    t_mask = hcd45_mask & (data[cd3] > gp['x_threshold']) & (data[cd19] < gp['y_threshold'])
                    n_t = t_mask.sum()
                    
                    for name, stats in quad_stats.items():
                        all_stats.append({'Population': name, 'Parent': 'hCD45+',
                                         'Count': stats['count'], '% Parent': round(stats['pct'], 1)})
                else:
                    t_mask = hcd45_mask
                    n_t = t_mask.sum()
                
                # Plot 7: CD4 vs CD8
                ax7 = fig.add_subplot(2, 4, 7)
                cd4 = channels['CD4']
                cd8 = channels['CD8']
                
                if cd4 and cd8 and 'cd4_cd8' in st.session_state.gates_params:
                    t_data = data[t_mask]
                    gp = st.session_state.gates_params['cd4_cd8']
                    
                    quad_stats = create_quadrant_plot(
                        ax7, t_data[cd4].values, t_data[cd8].values,
                        'CD4', 'CD8', f'T cells: {n_t:,}',
                        gp['x_threshold'], gp['y_threshold'],
                        ['DP', 'CD8+', 'DN', 'CD4+']
                    )
                    
                    # Créer le mask CD4+
                    cd4_mask = t_mask & (data[cd4] > gp['x_threshold']) & (data[cd8] < gp['y_threshold'])
                    n_cd4 = cd4_mask.sum()
                    
                    for name, stats in quad_stats.items():
                        all_stats.append({'Population': f'T-{name}', 'Parent': 'T cells',
                                         'Count': stats['count'], '% Parent': round(stats['pct'], 1)})
                else:
                    cd4_mask = t_mask
                    n_cd4 = cd4_mask.sum()
                
                # Plot 8: Treg (FoxP3 vs CD25)
                ax8 = fig.add_subplot(2, 4, 8)
                foxp3 = channels['FoxP3']
                cd25 = channels['CD25']
                
                if foxp3 and cd25 and 'treg' in st.session_state.gates_params:
                    cd4_data = data[cd4_mask]
                    gp = st.session_state.gates_params['treg']
                    
                    quad_stats = create_quadrant_plot(
                        ax8, cd4_data[foxp3].values, cd4_data[cd25].values,
                        'FoxP3', 'CD25', f'CD4+ T cells: {n_cd4:,}',
                        gp['x_threshold'], gp['y_threshold'],
                        ['Treg', 'CD25+', 'DN', 'FoxP3+']
                    )
                    
                    for name, stats in quad_stats.items():
                        all_stats.append({'Population': f'Treg-{name}', 'Parent': 'CD4+ T cells',
                                         'Count': stats['count'], '% Parent': round(stats['pct'], 1)})
                
                plt.tight_layout()
                st.pyplot(fig)
                st.session_state.fig = fig
                st.session_state.all_stats = all_stats
        
        # ==================== STATISTIQUES ====================
        
        if 'all_stats' in st.session_state:
            st.markdown("### 📋 Statistiques des Populations")
            
            stats_df = pd.DataFrame(st.session_state.all_stats)
            st.dataframe(stats_df, use_container_width=True)
            
            # Export
            col1, col2 = st.columns(2)
            
            with col1:
                if 'fig' in st.session_state:
                    buf = io.BytesIO()
                    st.session_state.fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        "📥 Télécharger PNG (300 DPI)",
                        buf,
                        f"{reader.filename}_gates.png",
                        "image/png",
                        use_container_width=True
                    )
            
            with col2:
                csv = stats_df.to_csv(index=False)
                st.download_button(
                    "📥 Télécharger CSV Statistiques",
                    csv,
                    f"{reader.filename}_stats.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    except Exception as e:
        st.error(f"Erreur: {e}")
        st.exception(e)

else:
    st.info("👆 Uploadez un fichier FCS pour commencer l'analyse")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🔬 <b>FACS Analysis - Gates Ajustables</b><br>
    Ajustez les seuils avec les sliders • Les graphiques se mettent à jour en temps réel
</div>
""", unsafe_allow_html=True)
