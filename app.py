#!/usr/bin/env python3
"""
Application Streamlit FACS - Style Publication avec Contrôles d'Axes
Version professionnelle avec density plots et ajustements d'axes
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import io
from datetime import datetime
import flowio
from scipy.stats import gaussian_kde

# Configuration
st.set_page_config(
    page_title="FACS Autogating - Publication",
    page_icon="🔬",
    layout="wide"
)

# CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    </style>
""", unsafe_allow_html=True)

class SimpleFCSReader:
    def __init__(self, fcs_path):
        self.fcs_path = fcs_path
        self.flow_data = flowio.FlowData(fcs_path)
        self.data = None
        self.channels = []
        self.load_data()
    
    def load_data(self):
        """Charge les données FCS en DataFrame"""
        events = self.flow_data.events
        n_channels = self.flow_data.channel_count
        
        if not isinstance(events, np.ndarray):
            events = np.array(events, dtype=np.float64)
        
        if events.ndim == 1:
            n_events = len(events) // n_channels
            events = events.reshape(n_events, n_channels)
        
        # Récupérer noms (majuscules ET minuscules)
        pnn_labels = []
        for i in range(1, n_channels + 1):
            pnn = self.flow_data.text.get(f'$P{i}N', None)
            if pnn is None or pnn == f'Channel_{i}':
                pnn = self.flow_data.text.get(f'p{i}n', f'Channel_{i}')
            pnn = pnn.strip() if isinstance(pnn, str) else pnn
            pnn_labels.append(pnn)
        
        self.channels = pnn_labels
        self.data = pd.DataFrame(events, columns=self.channels)
        return self.data
    
    def get_info(self):
        return {
            'event_count': len(self.data),
            'channel_count': len(self.channels),
            'channels': self.channels
        }

def gate_singlets_simple(data, fsc_a='FSC-A', fsc_h='FSC-H', threshold=1.5):
    if fsc_a not in data.columns or fsc_h not in data.columns:
        return pd.Series(True, index=data.index)
    
    ratio = data[fsc_h] / (data[fsc_a] + 1)
    median_ratio = ratio.median()
    mad = (ratio - median_ratio).abs().median()
    
    gate = (ratio > median_ratio - threshold * mad) & (ratio < median_ratio + threshold * mad)
    return gate

def gate_debris_simple(data, fsc='FSC-A', ssc='SSC-A', percentile=2):
    if fsc not in data.columns or ssc not in data.columns:
        return pd.Series(True, index=data.index)
    
    fsc_thresh = np.percentile(data[fsc], percentile)
    ssc_thresh = np.percentile(data[ssc], percentile)
    
    gate = (data[fsc] > fsc_thresh) & (data[ssc] > ssc_thresh)
    return gate

def gate_positive_simple(data, channel):
    if channel not in data.columns:
        return pd.Series(False, index=data.index)
    
    from sklearn.cluster import KMeans
    
    X = data[channel].values.reshape(-1, 1)
    q1, q99 = np.percentile(X, [1, 99])
    mask = (X >= q1) & (X <= q99)
    X_filtered = X[mask.flatten()]
    
    if len(X_filtered) < 100:
        return pd.Series(False, index=data.index)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_filtered)
    
    positive_cluster = np.argmax(kmeans.cluster_centers_)
    labels = kmeans.predict(X)
    gate = labels == positive_cluster
    
    return pd.Series(gate, index=data.index)

def create_publication_plot(data, x_channel, y_channel, gates, 
                           xlim=None, ylim=None, 
                           plot_type='scatter', log_scale=True,
                           show_density=True, show_contours=True):
    """Crée un plot style publication avec contrôles avancés"""
    
    # Figure de haute qualité
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    # Récupérer les données
    x_data = data[x_channel].values
    y_data = data[y_channel].values
    
    # Filtrer les valeurs invalides
    valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    
    # Échelle logarithmique si demandé
    if log_scale:
        x_data = np.log10(x_data + 1)
        y_data = np.log10(y_data + 1)
    
    # Limites automatiques si non spécifiées
    if xlim is None:
        xlim = (x_data.min(), x_data.max())
    if ylim is None:
        ylim = (y_data.min(), y_data.max())
    
    # Type de plot
    if plot_type == 'density' and show_density:
        # Density plot (hexbin)
        hb = ax.hexbin(x_data, y_data, gridsize=100, cmap='Blues', 
                      mincnt=1, bins='log', alpha=0.8)
        cb = plt.colorbar(hb, ax=ax, label='Densité (log)')
        
    elif plot_type == 'contour' and show_contours:
        # Contour plot
        try:
            # Créer une grille
            xi = np.linspace(xlim[0], xlim[1], 100)
            yi = np.linspace(ylim[0], ylim[1], 100)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # KDE
            positions = np.vstack([Xi.ravel(), Yi.ravel()])
            kernel = gaussian_kde(np.vstack([x_data, y_data]))
            Zi = np.reshape(kernel(positions).T, Xi.shape)
            
            # Contours
            levels = np.percentile(Zi, [10, 30, 50, 70, 90])
            cs = ax.contourf(Xi, Yi, Zi, levels=10, cmap='Blues', alpha=0.6)
            ax.contour(Xi, Yi, Zi, levels=levels, colors='black', linewidths=0.5, alpha=0.3)
            
        except:
            # Fallback sur scatter
            ax.scatter(x_data, y_data, s=1, c='steelblue', alpha=0.3, rasterized=True)
    
    else:
        # Scatter plot classique
        ax.scatter(x_data, y_data, s=0.5, c='lightgray', alpha=0.2, rasterized=True)
    
    # Overlay des gates si disponibles
    if gates:
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
        for idx, (gate_name, mask) in enumerate(gates.items()):
            color = colors[idx % len(colors)]
            mask_valid = mask[valid_mask]
            
            if mask_valid.sum() > 0:
                x_gated = x_data[mask_valid]
                y_gated = y_data[mask_valid]
                
                ax.scatter(x_gated, y_gated, s=1.5, c=color, alpha=0.6, 
                          label=f"{gate_name} ({mask.sum():,})", rasterized=True)
    
    # Style publication
    ax.set_xlabel(x_channel, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_channel, fontsize=14, fontweight='bold')
    ax.set_title(f'{x_channel} vs {y_channel}', fontsize=16, fontweight='bold', pad=20)
    
    # Limites d'axes
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Grille
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Légende
    if gates:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9, 
                 fancybox=True, shadow=True)
    
    # Bordures nettes
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # Ticks
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3)
    
    plt.tight_layout()
    
    return fig

def export_to_excel_complete(reader, gates, filename='facs_export_complet.xlsx'):
    """Export complet Excel"""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils.dataframe import dataframe_to_rows
    except:
        return None
    
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    
    # Statistiques
    ws_stats = wb.create_sheet("Statistiques")
    stats_data = []
    for gate_name, mask in gates.items():
        stats_data.append({
            'Population': gate_name,
            'Nombre': int(mask.sum()),
            'Pourcentage': round((mask.sum() / len(reader.data)) * 100, 2)
        })
    
    stats_df = pd.DataFrame(stats_data)
    for r_idx, row in enumerate(dataframe_to_rows(stats_df, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_stats.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    
    # Info fichier
    ws_info = wb.create_sheet("Info_Fichier")
    info_data = [
        ['Paramètre', 'Valeur'],
        ['Événements', len(reader.data)],
        ['Canaux', len(reader.channels)],
        ['Populations', len(gates)],
        ['Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    
    for r_idx, row in enumerate(info_data, 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_info.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
    
    # Canaux
    ws_channels = wb.create_sheet("Canaux")
    ws_channels.append(['Index', 'Canal'])
    for idx, ch in enumerate(reader.channels, 1):
        ws_channels.append([idx, ch])
    
    # Données échantillon
    ws_data = wb.create_sheet("Donnees_Echantillon")
    sample_data = reader.data.head(1000).copy()
    for gate_name, mask in gates.items():
        sample_data[f'Gate_{gate_name}'] = mask.head(1000).astype(int)
    
    for r_idx, row in enumerate(dataframe_to_rows(sample_data, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            ws_data.cell(row=r_idx, column=c_idx, value=value)
    
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output

# Interface
st.markdown('<h1 class="main-header">🔬 FACS - Style Publication</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    mode = st.radio("Mode", ["🔍 Analyse", "📊 Info Fichier"], index=0)
    
    st.markdown("---")
    st.markdown("### ⚙️ Gating")
    gate_singlets_option = st.checkbox("Gate Singlets", value=True)
    gate_debris_option = st.checkbox("Supprimer Débris", value=True)
    
    st.markdown("---")
    st.markdown("### 🎨 Visualisation")
    plot_type = st.selectbox("Type de plot", ["scatter", "density", "contour"], index=0)
    log_scale = st.checkbox("Échelle log", value=True)
    show_density = st.checkbox("Afficher densité", value=True)
    show_contours = st.checkbox("Afficher contours", value=False)

# Session state
if 'reader' not in st.session_state:
    st.session_state.reader = None
if 'gates' not in st.session_state:
    st.session_state.gates = {}
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'figures' not in st.session_state:
    st.session_state.figures = {}

if mode == "🔍 Analyse":
    st.markdown("### 📁 Upload Fichier FCS")
    
    uploaded_file = st.file_uploader("Sélectionner un fichier FCS", type=['fcs'])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        st.markdown(f'<div class="info-box">📄 <b>{uploaded_file.name}</b></div>', unsafe_allow_html=True)
        
        try:
            with st.spinner("Chargement..."):
                reader = SimpleFCSReader(tmp_path)
                st.session_state.reader = reader
            
            info = reader.get_info()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Événements", f"{info['event_count']:,}")
            with col2:
                st.metric("Canaux", info['channel_count'])
            with col3:
                st.metric("Taille", f"{uploaded_file.size / 1024 / 1024:.1f} MB")
            
            if st.button("🚀 Analyser", type="primary", use_container_width=True):
                with st.spinner("Analyse..."):
                    data = reader.data
                    gates = {}
                    
                    if gate_singlets_option:
                        fsc_a = [c for c in data.columns if 'FSC' in c.upper() and 'A' in c.upper()]
                        fsc_h = [c for c in data.columns if 'FSC' in c.upper() and 'H' in c.upper()]
                        
                        if fsc_a and fsc_h:
                            gates['singlets'] = gate_singlets_simple(data, fsc_a[0], fsc_h[0])
                            st.success(f"✅ Singlets : {gates['singlets'].sum():,}")
                    
                    if gate_debris_option:
                        fsc_a = [c for c in data.columns if 'FSC' in c.upper() and 'A' in c.upper()]
                        ssc_a = [c for c in data.columns if 'SSC' in c.upper() and 'A' in c.upper()]
                        
                        if fsc_a and ssc_a:
                            parent = gates.get('singlets', pd.Series(True, index=data.index))
                            debris_gate = gate_debris_simple(data, fsc_a[0], ssc_a[0])
                            gates['viable'] = parent & debris_gate
                            st.success(f"✅ Viables : {gates['viable'].sum():,}")
                    
                    marker_channels = [c for c in data.columns if any(m in c.upper() for m in ['CD', 'FITC', 'PE', 'APC', 'BV', 'AF'])]
                    
                    if marker_channels:
                        parent = gates.get('viable', gates.get('singlets', pd.Series(True, index=data.index)))
                        marker = marker_channels[0]
                        try:
                            marker_gate = gate_positive_simple(data[parent], marker)
                            full_gate = parent.copy()
                            full_gate[parent] = marker_gate
                            gates[f'{marker}_pos'] = full_gate
                            st.success(f"✅ {marker}+ : {full_gate.sum():,}")
                        except:
                            pass
                    
                    st.session_state.gates = gates
                    st.session_state.analysis_done = True
                    st.markdown('<div class="success-box">✅ Terminé !</div>', unsafe_allow_html=True)
            
            if st.session_state.analysis_done and st.session_state.gates:
                st.markdown("---")
                st.markdown("### 📊 Résultats")
                
                tab1, tab2, tab3 = st.tabs(["📈 Statistiques", "🎨 Visualisations", "💾 Export"])
                
                with tab1:
                    stats_data = []
                    for gate_name, mask in st.session_state.gates.items():
                        stats_data.append({
                            'Population': gate_name,
                            'Nombre': f"{mask.sum():,}",
                            '% Total': f"{(mask.sum() / len(reader.data)) * 100:.1f}%"
                        })
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)
                
                with tab2:
                    st.markdown("### 🎨 Visualisations Publication")
                    
                    all_channels = list(reader.channels)
                    
                    # Sélection canaux
                    col1, col2 = st.columns(2)
                    with col1:
                        x_channel = st.selectbox("Canal X", all_channels, index=0)
                    with col2:
                        y_channel = st.selectbox("Canal Y", all_channels, index=1 if len(all_channels) > 1 else 0)
                    
                    # Contrôles d'axes
                    st.markdown("#### 📏 Contrôles des Axes")
                    
                    # Calculer limites par défaut
                    if x_channel and y_channel:
                        x_data = reader.data[x_channel].values
                        y_data = reader.data[y_channel].values
                        
                        # Filtrer valeurs valides
                        x_valid = x_data[(x_data > 0) & np.isfinite(x_data)]
                        y_valid = y_data[(y_data > 0) & np.isfinite(y_data)]
                        
                        if log_scale:
                            x_valid = np.log10(x_valid + 1)
                            y_valid = np.log10(y_valid + 1)
                        
                        x_min_default = float(x_valid.min())
                        x_max_default = float(x_valid.max())
                        y_min_default = float(y_valid.min())
                        y_max_default = float(y_valid.max())
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            x_min = st.number_input("X min", value=x_min_default, format="%.2f", step=0.1)
                        with col2:
                            x_max = st.number_input("X max", value=x_max_default, format="%.2f", step=0.1)
                        with col3:
                            y_min = st.number_input("Y min", value=y_min_default, format="%.2f", step=0.1)
                        with col4:
                            y_max = st.number_input("Y max", value=y_max_default, format="%.2f", step=0.1)
                        
                        # Bouton reset
                        if st.button("🔄 Réinitialiser axes"):
                            st.rerun()
                        
                        st.markdown("---")
                        
                        # Générer le plot
                        fig = create_publication_plot(
                            reader.data, x_channel, y_channel, st.session_state.gates,
                            xlim=(x_min, x_max), ylim=(y_min, y_max),
                            plot_type=plot_type, log_scale=log_scale,
                            show_density=show_density, show_contours=show_contours
                        )
                        
                        st.pyplot(fig)
                        st.session_state.figures[f'plot_{x_channel}_{y_channel}'] = fig
                
                with tab3:
                    st.markdown("### 💾 Export")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 Excel Complet")
                        if st.button("Générer Excel", type="primary", use_container_width=True):
                            excel_data = export_to_excel_complete(reader, st.session_state.gates)
                            if excel_data:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                st.download_button(
                                    label="📥 Télécharger Excel",
                                    data=excel_data,
                                    file_name=f"FACS_Export_{timestamp}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                    
                    with col2:
                        st.markdown("#### 🖼️ Image Haute Résolution")
                        plot_keys = [k for k in st.session_state.figures.keys() if k.startswith('plot_')]
                        if plot_keys:
                            buf = io.BytesIO()
                            st.session_state.figures[plot_keys[0]].savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            buf.seek(0)
                            
                            st.download_button(
                                label="📥 Télécharger PNG (300 DPI)",
                                data=buf,
                                file_name=f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
        
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
            st.exception(e)

elif mode == "📊 Info Fichier":
    st.markdown("### 📊 Analyse des Canaux")
    
    uploaded_file = st.file_uploader("Fichier FCS", type=['fcs'], key="info")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            reader = SimpleFCSReader(tmp_path)
            info = reader.get_info()
            
            st.markdown(f"**Fichier** : {uploaded_file.name}")
            st.markdown(f"**Événements** : {info['event_count']:,}")
            st.markdown(f"**Canaux** : {info['channel_count']}")
            
            st.markdown("---")
            channels_df = pd.DataFrame({
                'Index': range(1, len(info['channels']) + 1),
                'Canal': info['channels']
            })
            st.dataframe(channels_df, use_container_width=True, height=400)
            
            csv = channels_df.to_csv(index=False)
            st.download_button(
                "📥 Exporter Liste",
                csv,
                f"canaux_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        except Exception as e:
            st.error(f"❌ {str(e)}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>🔬 <b>FACS Autogating - Style Publication</b></p>
    <p>Contrôles d'axes • Density plots • Export haute résolution</p>
</div>
""", unsafe_allow_html=True)
