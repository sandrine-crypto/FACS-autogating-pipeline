#!/usr/bin/env python3
"""
FACS Autogating - Version Complète
- Grille de visualisations sur une page
- Excel détaillé avec statistiques par marqueur
- Contrôles d'axes
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

# Configuration
st.set_page_config(
    page_title="FACS Autogating - Complet",
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
        self.filename = Path(fcs_path).stem
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
            'channels': self.channels,
            'filename': self.filename
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


def gate_positive_simple(data, channel, parent_gate=None):
    """Gating positif/négatif avec KMeans"""
    if channel not in data.columns:
        return pd.Series(False, index=data.index), 0, {}
    
    from sklearn.cluster import KMeans
    
    # Appliquer le gate parent si fourni
    if parent_gate is not None:
        working_data = data[parent_gate]
    else:
        working_data = data
    
    if len(working_data) < 100:
        return pd.Series(False, index=data.index), 0, {}
    
    X = working_data[channel].values.reshape(-1, 1)
    
    # Filtrer valeurs extrêmes
    q1, q99 = np.percentile(X, [1, 99])
    mask = (X >= q1) & (X <= q99)
    X_filtered = X[mask.flatten()]
    
    if len(X_filtered) < 100:
        return pd.Series(False, index=data.index), 0, {}
    
    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_filtered)
    
    positive_cluster = np.argmax(kmeans.cluster_centers_)
    labels = kmeans.predict(X)
    
    # Créer le gate complet
    gate = pd.Series(False, index=data.index)
    if parent_gate is not None:
        gate.loc[parent_gate] = (labels == positive_cluster)
    else:
        gate = pd.Series(labels == positive_cluster, index=data.index)
    
    # Statistiques
    pos_count = gate.sum()
    total_count = parent_gate.sum() if parent_gate is not None else len(data)
    percentage = (pos_count / total_count * 100) if total_count > 0 else 0
    
    # Stats détaillées
    stats = {
        'count': int(pos_count),
        'percentage': round(percentage, 2),
        'mean': round(float(working_data.loc[gate[parent_gate] if parent_gate is not None else gate, channel].mean()), 2) if pos_count > 0 else 0,
        'median': round(float(working_data.loc[gate[parent_gate] if parent_gate is not None else gate, channel].median()), 2) if pos_count > 0 else 0,
        'std': round(float(working_data.loc[gate[parent_gate] if parent_gate is not None else gate, channel].std()), 2) if pos_count > 0 else 0,
    }
    
    return gate, percentage, stats


def create_multi_plot_figure(data, channels_pairs, gates, log_scale=True, 
                             xlims=None, ylims=None, ncols=3):
    """Crée une figure avec plusieurs plots en grille"""
    
    n_plots = len(channels_pairs)
    ncols = min(ncols, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), dpi=120)
    
    # Aplatir les axes si nécessaire
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    for idx, (x_ch, y_ch) in enumerate(channels_pairs):
        ax = axes[idx]
        
        if x_ch not in data.columns or y_ch not in data.columns:
            ax.text(0.5, 0.5, f'Canal non trouvé', ha='center', va='center')
            ax.set_title(f'{x_ch} vs {y_ch}')
            continue
        
        x_data = data[x_ch].values.copy()
        y_data = data[y_ch].values.copy()
        
        # Filtrer valeurs invalides
        valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_plot = x_data[valid_mask]
        y_plot = y_data[valid_mask]
        
        if log_scale:
            x_plot = np.log10(x_plot + 1)
            y_plot = np.log10(y_plot + 1)
        
        # Plot de fond (densité)
        ax.hexbin(x_plot, y_plot, gridsize=50, cmap='Greys', mincnt=1, alpha=0.5)
        
        # Overlay des gates
        for g_idx, (gate_name, mask) in enumerate(gates.items()):
            color = colors[g_idx % len(colors)]
            
            mask_array = mask.values.astype(bool) if hasattr(mask, 'values') else np.array(mask, dtype=bool)
            mask_valid = mask_array[valid_mask].astype(bool)
            
            if mask_valid.sum() > 0:
                ax.scatter(x_plot[mask_valid], y_plot[mask_valid], 
                          s=1, c=color, alpha=0.4, label=f"{gate_name}", rasterized=True)
        
        # Limites d'axes
        if xlims and idx < len(xlims) and xlims[idx]:
            ax.set_xlim(xlims[idx])
        if ylims and idx < len(ylims) and ylims[idx]:
            ax.set_ylim(ylims[idx])
        
        ax.set_xlabel(x_ch, fontsize=10)
        ax.set_ylabel(y_ch, fontsize=10)
        ax.set_title(f'{x_ch} vs {y_ch}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if g_idx == 0:  # Légende seulement sur le premier plot
            ax.legend(loc='upper right', fontsize=7, markerscale=3)
    
    # Masquer les axes inutilisés
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def export_complete_excel(reader, gates, marker_stats, filename):
    """Export Excel complet avec statistiques détaillées par marqueur"""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils.dataframe import dataframe_to_rows
    except ImportError:
        return None
    
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    
    # Styles
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill_blue = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_fill_green = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
    header_fill_orange = PatternFill(start_color="ED7D31", end_color="ED7D31", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # ==================== FEUILLE 1 : RÉSUMÉ ====================
    ws_summary = wb.create_sheet("Résumé")
    
    summary_data = [
        ['RÉSUMÉ DE L\'ANALYSE FACS'],
        [''],
        ['Paramètre', 'Valeur'],
        ['Fichier', reader.filename],
        ['Date d\'analyse', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ['Événements totaux', len(reader.data)],
        ['Nombre de canaux', len(reader.channels)],
        ['Populations identifiées', len(gates)],
        [''],
    ]
    
    for r_idx, row in enumerate(summary_data, 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_summary.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:
                cell.font = Font(bold=True, size=14, color="4472C4")
            elif r_idx == 3:
                cell.font = header_font
                cell.fill = header_fill_blue
    
    # ==================== FEUILLE 2 : STATISTIQUES POPULATIONS ====================
    ws_stats = wb.create_sheet("Statistiques_Populations")
    
    headers = ['Population', 'Nombre', '% du Total', '% du Parent']
    ws_stats.append(headers)
    
    for c_idx, header in enumerate(headers, 1):
        cell = ws_stats.cell(row=1, column=c_idx)
        cell.font = header_font
        cell.fill = header_fill_blue
        cell.alignment = Alignment(horizontal='center')
    
    total_events = len(reader.data)
    parent_count = total_events
    
    for gate_name, mask in gates.items():
        count = int(mask.sum())
        pct_total = round((count / total_events) * 100, 2)
        pct_parent = round((count / parent_count) * 100, 2) if parent_count > 0 else 0
        
        ws_stats.append([gate_name, count, f"{pct_total}%", f"{pct_parent}%"])
        parent_count = count  # Pour hiérarchie
    
    # ==================== FEUILLE 3 : STATISTIQUES PAR MARQUEUR ====================
    ws_markers = wb.create_sheet("Statistiques_Marqueurs")
    
    marker_headers = ['Marqueur', 'Population', 'Count', '% Positif', 'MFI Mean', 'MFI Median', 'MFI Std']
    ws_markers.append(marker_headers)
    
    for c_idx, header in enumerate(marker_headers, 1):
        cell = ws_markers.cell(row=1, column=c_idx)
        cell.font = header_font
        cell.fill = header_fill_green
        cell.alignment = Alignment(horizontal='center')
    
    for marker, stats in marker_stats.items():
        ws_markers.append([
            marker,
            stats.get('parent', 'viable'),
            stats.get('count', 0),
            f"{stats.get('percentage', 0)}%",
            stats.get('mean', 0),
            stats.get('median', 0),
            stats.get('std', 0)
        ])
    
    # ==================== FEUILLE 4 : TABLEAU RÉCAPITULATIF MARQUEURS ====================
    ws_recap = wb.create_sheet("Recap_Marqueurs")
    
    # Créer un tableau pivot-like
    ws_recap.append(['TABLEAU RÉCAPITULATIF DES MARQUEURS'])
    ws_recap.append([''])
    
    recap_headers = ['Marqueur', 'Count Positif', '% Positif', 'MFI']
    ws_recap.append(recap_headers)
    
    for c_idx, header in enumerate(recap_headers, 1):
        cell = ws_recap.cell(row=3, column=c_idx)
        cell.font = header_font
        cell.fill = header_fill_orange
        cell.alignment = Alignment(horizontal='center')
    
    row_idx = 4
    for marker, stats in marker_stats.items():
        ws_recap.cell(row=row_idx, column=1, value=marker)
        ws_recap.cell(row=row_idx, column=2, value=stats.get('count', 0))
        ws_recap.cell(row=row_idx, column=3, value=f"{stats.get('percentage', 0)}%")
        ws_recap.cell(row=row_idx, column=4, value=stats.get('mean', 0))
        row_idx += 1
    
    # ==================== FEUILLE 5 : LISTE DES CANAUX ====================
    ws_channels = wb.create_sheet("Canaux")
    
    ws_channels.append(['Index', 'Canal', 'Type'])
    for c_idx in range(1, 4):
        cell = ws_channels.cell(row=1, column=c_idx)
        cell.font = header_font
        cell.fill = header_fill_blue
    
    for idx, ch in enumerate(reader.channels, 1):
        ch_type = 'Scatter' if any(s in ch.upper() for s in ['FSC', 'SSC']) else 'Fluorescence'
        ws_channels.append([idx, ch, ch_type])
    
    # ==================== FEUILLE 6 : DONNÉES ÉCHANTILLON ====================
    ws_data = wb.create_sheet("Donnees_Echantillon")
    
    sample_data = reader.data.head(500).copy()
    for gate_name, mask in gates.items():
        sample_data[f'Gate_{gate_name}'] = mask.head(500).astype(int)
    
    for r_idx, row in enumerate(dataframe_to_rows(sample_data, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_data.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    
    # Ajuster largeurs de colonnes
    for ws in wb.worksheets:
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output


# ==================== INTERFACE STREAMLIT ====================

st.markdown('<h1 class="main-header">🔬 FACS Autogating - Analyse Complète</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Configuration")
    
    st.markdown("### ⚙️ Gating")
    gate_singlets = st.checkbox("Gate Singlets", value=True)
    gate_debris = st.checkbox("Supprimer Débris", value=True)
    
    st.markdown("### 🎨 Visualisation")
    log_scale = st.checkbox("Échelle log", value=True)
    ncols = st.slider("Colonnes par ligne", 2, 4, 3)
    
    st.markdown("### 📊 Marqueurs à analyser")
    analyze_all_markers = st.checkbox("Analyser tous les marqueurs", value=True)

# Session state
if 'reader' not in st.session_state:
    st.session_state.reader = None
if 'gates' not in st.session_state:
    st.session_state.gates = {}
if 'marker_stats' not in st.session_state:
    st.session_state.marker_stats = {}
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Upload
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
        
        # Bouton Analyser
        if st.button("🚀 Analyser", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                data = reader.data
                gates = {}
                marker_stats = {}
                
                # Gate Singlets
                if gate_singlets:
                    fsc_a = [c for c in data.columns if 'FSC' in c.upper() and '-A' in c.upper()]
                    fsc_h = [c for c in data.columns if 'FSC' in c.upper() and '-H' in c.upper()]
                    
                    if fsc_a and fsc_h:
                        gates['singlets'] = gate_singlets_simple(data, fsc_a[0], fsc_h[0])
                        st.success(f"✅ Singlets : {gates['singlets'].sum():,} ({gates['singlets'].sum()/len(data)*100:.1f}%)")
                
                # Gate Débris
                if gate_debris:
                    fsc_a = [c for c in data.columns if 'FSC' in c.upper() and '-A' in c.upper()]
                    ssc_a = [c for c in data.columns if 'SSC' in c.upper() and '-A' in c.upper()]
                    
                    if fsc_a and ssc_a:
                        parent = gates.get('singlets', pd.Series(True, index=data.index))
                        debris_gate = gate_debris_simple(data, fsc_a[0], ssc_a[0])
                        gates['viable'] = parent & debris_gate
                        st.success(f"✅ Viables : {gates['viable'].sum():,} ({gates['viable'].sum()/len(data)*100:.1f}%)")
                
                # Détecter et analyser TOUS les marqueurs
                marker_keywords = ['CD', 'FITC', 'PE', 'APC', 'BV', 'AF', 'BUV', 'eFluor', 
                                  'Pacific', 'Alexa', 'PerCP', 'Live', 'Dead', 'Nova']
                
                marker_channels = [c for c in data.columns 
                                  if any(m.upper() in c.upper() for m in marker_keywords)
                                  and not any(s in c.upper() for s in ['FSC', 'SSC', 'TIME'])]
                
                if marker_channels:
                    st.info(f"🔍 {len(marker_channels)} marqueurs détectés")
                    
                    parent_gate = gates.get('viable', gates.get('singlets', pd.Series(True, index=data.index)))
                    
                    progress_bar = st.progress(0)
                    
                    for i, marker in enumerate(marker_channels):
                        try:
                            gate, pct, stats = gate_positive_simple(data, marker, parent_gate)
                            
                            if stats:
                                gates[f'{marker}_pos'] = gate
                                stats['parent'] = 'viable' if 'viable' in gates else 'singlets'
                                marker_stats[marker] = stats
                        except Exception as e:
                            pass
                        
                        progress_bar.progress((i + 1) / len(marker_channels))
                    
                    progress_bar.empty()
                    st.success(f"✅ {len(marker_stats)} marqueurs analysés")
                
                st.session_state.gates = gates
                st.session_state.marker_stats = marker_stats
                st.session_state.analysis_done = True
                
                st.markdown('<div class="success-box">✅ Analyse terminée !</div>', unsafe_allow_html=True)
        
        # Résultats
        if st.session_state.analysis_done:
            st.markdown("---")
            st.markdown("### 📊 Résultats")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Statistiques", 
                "🎨 Visualisations (Grille)", 
                "📊 Marqueurs Détaillés",
                "💾 Export"
            ])
            
            # TAB 1 : STATISTIQUES
            with tab1:
                st.markdown("#### Populations Identifiées")
                
                stats_data = []
                total = len(reader.data)
                
                for gate_name, mask in st.session_state.gates.items():
                    if '_pos' not in gate_name:  # Seulement gates principaux
                        stats_data.append({
                            'Population': gate_name,
                            'Nombre': int(mask.sum()),
                            '% Total': f"{(mask.sum() / total) * 100:.1f}%"
                        })
                
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
                
                # Graphique barres
                if stats_data:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    pops = [s['Population'] for s in stats_data]
                    counts = [s['Nombre'] for s in stats_data]
                    ax.barh(pops, counts, color='steelblue')
                    ax.set_xlabel('Nombre d\'événements')
                    ax.set_title('Populations', fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # TAB 2 : VISUALISATIONS GRILLE
            with tab2:
                st.markdown("#### 🎨 Grille de Visualisations")
                
                # Définir les paires de canaux à afficher
                all_channels = reader.channels
                
                # Canaux scatter
                fsc_a = [c for c in all_channels if 'FSC' in c.upper() and '-A' in c.upper()]
                fsc_h = [c for c in all_channels if 'FSC' in c.upper() and '-H' in c.upper()]
                ssc_a = [c for c in all_channels if 'SSC' in c.upper() and '-A' in c.upper()]
                
                # Paires par défaut
                default_pairs = []
                if fsc_a and ssc_a:
                    default_pairs.append((fsc_a[0], ssc_a[0]))
                if fsc_a and fsc_h:
                    default_pairs.append((fsc_a[0], fsc_h[0]))
                
                # Ajouter quelques paires de marqueurs
                markers = [c for c in all_channels if any(m in c.upper() for m in ['CD', 'BV', 'AF', 'PE', 'APC'])]
                for i in range(min(4, len(markers))):
                    if fsc_a:
                        default_pairs.append((markers[i], fsc_a[0]))
                
                # Sélection personnalisée
                st.markdown("##### Sélection des graphiques")
                
                col1, col2 = st.columns(2)
                with col1:
                    n_plots = st.slider("Nombre de graphiques", 1, 12, min(6, len(default_pairs)))
                
                # Contrôles d'axes globaux
                st.markdown("##### Contrôles des axes")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    x_min = st.number_input("X min (global)", value=0.0, step=0.5)
                with col2:
                    x_max = st.number_input("X max (global)", value=6.0, step=0.5)
                with col3:
                    y_min = st.number_input("Y min (global)", value=0.0, step=0.5)
                with col4:
                    y_max = st.number_input("Y max (global)", value=6.0, step=0.5)
                
                use_custom_limits = st.checkbox("Appliquer limites personnalisées", value=False)
                
                # Générer la grille
                if st.button("📊 Générer la grille de visualisations", type="primary"):
                    with st.spinner("Génération des graphiques..."):
                        pairs_to_plot = default_pairs[:n_plots]
                        
                        xlims = [(x_min, x_max)] * n_plots if use_custom_limits else None
                        ylims = [(y_min, y_max)] * n_plots if use_custom_limits else None
                        
                        # Gates à afficher (seulement principaux)
                        display_gates = {k: v for k, v in st.session_state.gates.items() 
                                        if k in ['singlets', 'viable']}
                        
                        fig = create_multi_plot_figure(
                            reader.data, pairs_to_plot, display_gates,
                            log_scale=log_scale, xlims=xlims, ylims=ylims, ncols=ncols
                        )
                        
                        st.pyplot(fig)
                        st.session_state.grid_figure = fig
                
                # Export de la grille
                if 'grid_figure' in st.session_state:
                    buf = io.BytesIO()
                    st.session_state.grid_figure.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        "📥 Télécharger Grille (PNG 300 DPI)",
                        buf,
                        f"grille_facs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        "image/png",
                        use_container_width=True
                    )
            
            # TAB 3 : MARQUEURS DÉTAILLÉS
            with tab3:
                st.markdown("#### 📊 Statistiques Détaillées par Marqueur")
                
                if st.session_state.marker_stats:
                    # Tableau complet
                    marker_df = pd.DataFrame([
                        {
                            'Marqueur': marker,
                            'Count': stats['count'],
                            '% Positif': f"{stats['percentage']}%",
                            'MFI Mean': stats['mean'],
                            'MFI Median': stats['median'],
                            'MFI Std': stats['std']
                        }
                        for marker, stats in st.session_state.marker_stats.items()
                    ])
                    
                    st.dataframe(marker_df, use_container_width=True, height=400)
                    
                    # Graphique des pourcentages
                    st.markdown("##### Distribution des % Positifs")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    markers = list(st.session_state.marker_stats.keys())
                    percentages = [st.session_state.marker_stats[m]['percentage'] for m in markers]
                    
                    # Trier par pourcentage
                    sorted_idx = np.argsort(percentages)[::-1]
                    markers_sorted = [markers[i] for i in sorted_idx]
                    pct_sorted = [percentages[i] for i in sorted_idx]
                    
                    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(markers_sorted)))
                    
                    ax.barh(range(len(markers_sorted)), pct_sorted, color=colors)
                    ax.set_yticks(range(len(markers_sorted)))
                    ax.set_yticklabels(markers_sorted, fontsize=8)
                    ax.set_xlabel('% Positif', fontsize=11)
                    ax.set_title('Pourcentage de Cellules Positives par Marqueur', fontsize=13, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Aucun marqueur analysé. Lancez l'analyse.")
            
            # TAB 4 : EXPORT
            with tab4:
                st.markdown("#### 💾 Export Complet")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 📊 Excel Complet")
                    st.info("Inclut :\n"
                            "- Résumé\n"
                            "- Statistiques populations\n"
                            "- **Statistiques par marqueur** (%, MFI)\n"
                            "- Tableau récapitulatif\n"
                            "- Liste des canaux\n"
                            "- Données échantillon")
                    
                    if st.button("Générer Excel Complet", type="primary", use_container_width=True):
                        with st.spinner("Génération..."):
                            excel_data = export_complete_excel(
                                reader, 
                                st.session_state.gates, 
                                st.session_state.marker_stats,
                                reader.filename
                            )
                            
                            if excel_data:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                st.download_button(
                                    "📥 Télécharger Excel",
                                    excel_data,
                                    f"FACS_Analyse_{reader.filename}_{timestamp}.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                
                with col2:
                    st.markdown("##### 📋 CSV Marqueurs")
                    
                    if st.session_state.marker_stats:
                        csv_data = pd.DataFrame([
                            {
                                'Echantillon': reader.filename,
                                'Marqueur': marker,
                                'Count': stats['count'],
                                'Percentage': stats['percentage'],
                                'MFI_Mean': stats['mean'],
                                'MFI_Median': stats['median'],
                                'MFI_Std': stats['std']
                            }
                            for marker, stats in st.session_state.marker_stats.items()
                        ])
                        
                        csv = csv_data.to_csv(index=False)
                        
                        st.download_button(
                            "📥 Télécharger CSV Marqueurs",
                            csv,
                            f"marqueurs_{reader.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
    
    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>🔬 <b>FACS Autogating - Analyse Complète</b></p>
    <p>Grille de visualisations • Statistiques par marqueur • Export Excel détaillé</p>
</div>
""", unsafe_allow_html=True)
