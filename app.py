#!/usr/bin/env python3
"""
FACS Autogating - Version Complète avec Identification des Marqueurs
- Détection automatique des marqueurs (CD4, CD8, CD3, FoxP3, etc.)
- Grille de visualisations avec noms de marqueurs
- Excel détaillé avec mapping Canal → Marqueur
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
import re

# Configuration
st.set_page_config(
    page_title="FACS Autogating - Marqueurs",
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


# Liste des marqueurs connus
KNOWN_MARKERS = [
    # Marqueurs T cells
    'CD3', 'CD4', 'CD8', 'CD8a', 'CD8b',
    # Marqueurs B cells
    'CD19', 'CD20', 'CD21', 'CD22', 'CD27',
    # Marqueurs NK
    'CD16', 'CD56', 'CD57',
    # Marqueurs d'activation
    'CD25', 'CD69', 'CD38', 'CD137', 'CD154', 'CD107a', 'CD107b',
    # Marqueurs mémoire/naïf
    'CD45', 'CD45RA', 'CD45RO', 'CD62L', 'CCR7', 'CD127',
    # Marqueurs Treg
    'FoxP3', 'FOXP3', 'CD127',
    # Checkpoint/exhaustion
    'PD-1', 'PD1', 'PDCD1', 'CTLA-4', 'CTLA4', 'TIM-3', 'TIM3', 
    'LAG-3', 'LAG3', 'TIGIT', 'BTLA',
    # Marqueurs fonctionnels
    'IFN-g', 'IFNg', 'IFN-gamma', 'TNF-a', 'TNFa', 'TNF-alpha',
    'IL-2', 'IL2', 'IL-4', 'IL4', 'IL-10', 'IL10', 'IL-17', 'IL17',
    'Granzyme', 'GranzymeB', 'GzmB', 'Perforin', 'Ki-67', 'Ki67',
    # Autres
    'HLA-DR', 'HLADR', 'HLA-A', 'HLA-B', 'HLA-C',
    'CD161', 'CD183', 'CXCR3', 'CD185', 'CXCR5', 'CD194', 'CCR4',
    'CD196', 'CCR6', 'CD197',
    # Viabilité
    'Live', 'Dead', 'Viability', 'Viab', '7-AAD', '7AAD', 'PI',
    # Lignée
    'CD14', 'CD33', 'CD11b', 'CD11c', 'CD123', 'CD303',
    # Autres marqueurs courants
    'PDL1', 'PD-L1', 'LLT1', 'NKG2D', 'NKp46',
]


def extract_marker_from_text(text):
    """Extrait le nom du marqueur depuis un texte (PnS ou PnN)"""
    if not text or not isinstance(text, str):
        return None
    
    text_upper = text.upper()
    
    # Patterns spécifiques pour extraire le marqueur
    # Ex: "hCD4 : BV650 - Area" -> CD4
    # Ex: "FoxP3 : eFluor450 - Area" -> FoxP3
    
    # Pattern 1: hXXX ou mXXX (human/mouse prefix)
    match = re.search(r'[hm]?(CD\d+[a-zA-Z]?)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 2: Chercher les marqueurs connus
    for marker in KNOWN_MARKERS:
        # Recherche exacte avec limites de mots
        pattern = r'\b' + re.escape(marker) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            return marker
    
    # Pattern 3: Extraire avant le ":"
    if ':' in text:
        before_colon = text.split(':')[0].strip()
        # Nettoyer le préfixe h/m
        cleaned = re.sub(r'^[hm]', '', before_colon, flags=re.IGNORECASE)
        if cleaned and len(cleaned) > 1:
            return cleaned
    
    return None


class SimpleFCSReader:
    def __init__(self, fcs_path):
        self.fcs_path = fcs_path
        self.flow_data = flowio.FlowData(fcs_path)
        self.data = None
        self.channels = []
        self.channel_info = {}  # Mapping canal -> {marker, fluorochrome, pns}
        self.filename = Path(fcs_path).stem
        self.load_data()
    
    def load_data(self):
        """Charge les données FCS en DataFrame avec extraction des marqueurs"""
        events = self.flow_data.events
        n_channels = self.flow_data.channel_count
        
        if not isinstance(events, np.ndarray):
            events = np.array(events, dtype=np.float64)
        
        if events.ndim == 1:
            n_events = len(events) // n_channels
            events = events.reshape(n_events, n_channels)
        
        pnn_labels = []
        for i in range(1, n_channels + 1):
            # Récupérer PnN (nom du canal)
            pnn = self.flow_data.text.get(f'$P{i}N', None)
            if pnn is None or pnn == f'Channel_{i}':
                pnn = self.flow_data.text.get(f'p{i}n', f'Channel_{i}')
            pnn = pnn.strip() if isinstance(pnn, str) else str(pnn)
            
            # Récupérer PnS (nom du stain/marqueur)
            pns = self.flow_data.text.get(f'$P{i}S', None)
            if pns is None:
                pns = self.flow_data.text.get(f'p{i}s', '')
            pns = pns.strip() if isinstance(pns, str) else str(pns)
            
            # Extraire le marqueur
            marker = extract_marker_from_text(pns) or extract_marker_from_text(pnn)
            
            # Extraire le fluorochrome (depuis PnN généralement)
            fluorochrome = pnn.replace('-A', '').replace('-H', '').replace('-W', '').strip()
            
            # Stocker les infos
            self.channel_info[pnn] = {
                'index': i,
                'pnn': pnn,
                'pns': pns,
                'marker': marker,
                'fluorochrome': fluorochrome,
                'display_name': f"{marker} ({fluorochrome})" if marker else pnn
            }
            
            pnn_labels.append(pnn)
        
        self.channels = pnn_labels
        self.data = pd.DataFrame(events, columns=self.channels)
        return self.data
    
    def get_marker_for_channel(self, channel):
        """Retourne le marqueur pour un canal donné"""
        if channel in self.channel_info:
            return self.channel_info[channel].get('marker', None)
        return None
    
    def get_display_name(self, channel):
        """Retourne le nom d'affichage (Marqueur + Fluorochrome)"""
        if channel in self.channel_info:
            return self.channel_info[channel].get('display_name', channel)
        return channel
    
    def get_channels_with_markers(self):
        """Retourne la liste des canaux qui ont un marqueur identifié"""
        return [ch for ch, info in self.channel_info.items() if info.get('marker')]
    
    def get_info(self):
        return {
            'event_count': len(self.data),
            'channel_count': len(self.channels),
            'channels': self.channels,
            'filename': self.filename,
            'channel_info': self.channel_info
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
    
    if parent_gate is not None:
        working_data = data[parent_gate]
    else:
        working_data = data
    
    if len(working_data) < 100:
        return pd.Series(False, index=data.index), 0, {}
    
    X = working_data[channel].values.reshape(-1, 1)
    
    q1, q99 = np.percentile(X, [1, 99])
    mask = (X >= q1) & (X <= q99)
    X_filtered = X[mask.flatten()]
    
    if len(X_filtered) < 100:
        return pd.Series(False, index=data.index), 0, {}
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_filtered)
    
    positive_cluster = np.argmax(kmeans.cluster_centers_)
    labels = kmeans.predict(X)
    
    gate = pd.Series(False, index=data.index)
    if parent_gate is not None:
        gate.loc[parent_gate] = (labels == positive_cluster)
    else:
        gate = pd.Series(labels == positive_cluster, index=data.index)
    
    pos_count = gate.sum()
    total_count = parent_gate.sum() if parent_gate is not None else len(data)
    percentage = (pos_count / total_count * 100) if total_count > 0 else 0
    
    # Calculer MFI sur les cellules positives
    try:
        if parent_gate is not None:
            positive_cells = working_data.loc[gate[parent_gate], channel]
        else:
            positive_cells = working_data.loc[gate, channel]
        
        stats = {
            'count': int(pos_count),
            'percentage': round(percentage, 2),
            'mean': round(float(positive_cells.mean()), 2) if len(positive_cells) > 0 else 0,
            'median': round(float(positive_cells.median()), 2) if len(positive_cells) > 0 else 0,
            'std': round(float(positive_cells.std()), 2) if len(positive_cells) > 0 else 0,
        }
    except:
        stats = {
            'count': int(pos_count),
            'percentage': round(percentage, 2),
            'mean': 0, 'median': 0, 'std': 0
        }
    
    return gate, percentage, stats


def create_multi_plot_figure(data, channels_pairs, gates, reader, log_scale=True, 
                             xlims=None, ylims=None, ncols=3):
    """Crée une figure avec plusieurs plots en grille avec noms de marqueurs"""
    
    n_plots = len(channels_pairs)
    if n_plots == 0:
        return None
    
    ncols = min(ncols, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), dpi=120)
    
    if n_plots == 1:
        axes = np.array([axes])
    if nrows == 1 and ncols > 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    for idx, (x_ch, y_ch) in enumerate(channels_pairs):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        if x_ch not in data.columns or y_ch not in data.columns:
            ax.text(0.5, 0.5, f'Canal non trouvé', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{x_ch} vs {y_ch}')
            continue
        
        x_data = data[x_ch].values.copy()
        y_data = data[y_ch].values.copy()
        
        valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_plot = x_data[valid_mask]
        y_plot = y_data[valid_mask]
        
        if log_scale:
            x_plot = np.log10(x_plot + 1)
            y_plot = np.log10(y_plot + 1)
        
        # Plot de fond (densité)
        if len(x_plot) > 0:
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
        
        # Labels avec noms de marqueurs
        x_display = reader.get_display_name(x_ch)
        y_display = reader.get_display_name(y_ch)
        
        ax.set_xlabel(x_display, fontsize=9, fontweight='bold')
        ax.set_ylabel(y_display, fontsize=9, fontweight='bold')
        ax.set_title(f'{x_display}\nvs {y_display}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if idx == 0 and gates:
            ax.legend(loc='upper right', fontsize=7, markerscale=3)
    
    # Masquer les axes inutilisés
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def export_complete_excel(reader, gates, marker_stats, filename):
    """Export Excel complet avec mapping Canal → Marqueur"""
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
    header_fill_purple = PatternFill(start_color="7030A0", end_color="7030A0", fill_type="solid")
    
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
        ['Marqueurs identifiés', len([c for c in reader.channel_info.values() if c.get('marker')])],
        ['Populations identifiées', len(gates)],
    ]
    
    for r_idx, row in enumerate(summary_data, 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_summary.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:
                cell.font = Font(bold=True, size=14, color="4472C4")
            elif r_idx == 3:
                cell.font = header_font
                cell.fill = header_fill_blue
    
    # ==================== FEUILLE 2 : MAPPING CANAUX → MARQUEURS ====================
    ws_mapping = wb.create_sheet("Mapping_Canaux_Marqueurs")
    
    mapping_headers = ['Index', 'Canal (Fluorochrome)', 'Marqueur', 'Description Complète', 'Type']
    ws_mapping.append(mapping_headers)
    
    for c_idx, header in enumerate(mapping_headers, 1):
        cell = ws_mapping.cell(row=1, column=c_idx)
        cell.font = header_font
        cell.fill = header_fill_purple
        cell.alignment = Alignment(horizontal='center')
    
    for ch, info in reader.channel_info.items():
        marker = info.get('marker', '-')
        pns = info.get('pns', '')
        
        # Déterminer le type
        if 'FSC' in ch.upper() or 'SSC' in ch.upper():
            ch_type = 'Scatter'
        elif 'TIME' in ch.upper():
            ch_type = 'Time'
        elif marker and marker != '-':
            ch_type = 'Marqueur'
        else:
            ch_type = 'Fluorescence'
        
        ws_mapping.append([
            info.get('index', ''),
            ch,
            marker if marker else '-',
            pns,
            ch_type
        ])
    
    # ==================== FEUILLE 3 : STATISTIQUES PAR MARQUEUR ====================
    ws_markers = wb.create_sheet("Statistiques_Marqueurs")
    
    marker_headers = ['Marqueur', 'Canal', '% Positif', 'Count', 'MFI Mean', 'MFI Median', 'MFI Std', 'Population Parent']
    ws_markers.append(marker_headers)
    
    for c_idx, header in enumerate(marker_headers, 1):
        cell = ws_markers.cell(row=1, column=c_idx)
        cell.font = header_font
        cell.fill = header_fill_green
        cell.alignment = Alignment(horizontal='center')
    
    for channel, stats in marker_stats.items():
        marker = reader.get_marker_for_channel(channel)
        ws_markers.append([
            marker if marker else channel,
            channel,
            f"{stats.get('percentage', 0)}%",
            stats.get('count', 0),
            stats.get('mean', 0),
            stats.get('median', 0),
            stats.get('std', 0),
            stats.get('parent', 'viable')
        ])
    
    # ==================== FEUILLE 4 : TABLEAU RÉCAPITULATIF (FORMAT PUBLICATION) ====================
    ws_recap = wb.create_sheet("Tableau_Publication")
    
    ws_recap.append(['TABLEAU RÉCAPITULATIF - FORMAT PUBLICATION'])
    ws_recap.cell(row=1, column=1).font = Font(bold=True, size=14, color="4472C4")
    ws_recap.append([''])
    ws_recap.append(['Échantillon:', reader.filename])
    ws_recap.append(['Date:', datetime.now().strftime("%Y-%m-%d")])
    ws_recap.append([''])
    
    recap_headers = ['Marqueur', 'Fluorochrome', '% Positif', 'MFI']
    ws_recap.append(recap_headers)
    
    for c_idx, header in enumerate(recap_headers, 1):
        cell = ws_recap.cell(row=6, column=c_idx)
        cell.font = header_font
        cell.fill = header_fill_orange
        cell.alignment = Alignment(horizontal='center')
    
    row_idx = 7
    for channel, stats in marker_stats.items():
        marker = reader.get_marker_for_channel(channel)
        fluoro = reader.channel_info.get(channel, {}).get('fluorochrome', channel)
        
        ws_recap.cell(row=row_idx, column=1, value=marker if marker else '-')
        ws_recap.cell(row=row_idx, column=2, value=fluoro)
        ws_recap.cell(row=row_idx, column=3, value=f"{stats.get('percentage', 0)}%")
        ws_recap.cell(row=row_idx, column=4, value=stats.get('mean', 0))
        row_idx += 1
    
    # ==================== FEUILLE 5 : STATISTIQUES POPULATIONS ====================
    ws_stats = wb.create_sheet("Statistiques_Populations")
    
    pop_headers = ['Population', 'Nombre', '% du Total']
    ws_stats.append(pop_headers)
    
    for c_idx, header in enumerate(pop_headers, 1):
        cell = ws_stats.cell(row=1, column=c_idx)
        cell.font = header_font
        cell.fill = header_fill_blue
    
    total_events = len(reader.data)
    for gate_name, mask in gates.items():
        if '_pos' not in gate_name:
            count = int(mask.sum())
            pct = round((count / total_events) * 100, 2)
            ws_stats.append([gate_name, count, f"{pct}%"])
    
    # ==================== FEUILLE 6 : DONNÉES ÉCHANTILLON ====================
    ws_data = wb.create_sheet("Donnees_Echantillon")
    
    # Créer les en-têtes avec marqueurs
    sample_data = reader.data.head(500).copy()
    
    # Renommer les colonnes avec les marqueurs
    new_columns = []
    for col in sample_data.columns:
        marker = reader.get_marker_for_channel(col)
        if marker:
            new_columns.append(f"{marker} ({col})")
        else:
            new_columns.append(col)
    sample_data.columns = new_columns
    
    # Ajouter colonnes de gates
    for gate_name, mask in gates.items():
        if '_pos' not in gate_name:
            sample_data[f'Gate_{gate_name}'] = mask.head(500).astype(int)
    
    for r_idx, row in enumerate(dataframe_to_rows(sample_data, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_data.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:
                cell.font = Font(bold=True, size=9)
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
            adjusted_width = min(max_length + 2, 40)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output


# ==================== INTERFACE STREAMLIT ====================

st.markdown('<h1 class="main-header">🔬 FACS Autogating - Identification des Marqueurs</h1>', unsafe_allow_html=True)
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
    
    st.markdown("### 📊 Analyse")
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
        with st.spinner("Chargement et identification des marqueurs..."):
            reader = SimpleFCSReader(tmp_path)
            st.session_state.reader = reader
        
        info = reader.get_info()
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Événements", f"{info['event_count']:,}")
        with col2:
            st.metric("Canaux", info['channel_count'])
        with col3:
            markers_found = len([c for c in info['channel_info'].values() if c.get('marker')])
            st.metric("Marqueurs ID", markers_found)
        with col4:
            st.metric("Taille", f"{uploaded_file.size / 1024 / 1024:.1f} MB")
        
        # Afficher les marqueurs identifiés
        with st.expander("🔬 Marqueurs Identifiés", expanded=True):
            marker_list = []
            for ch, info_ch in reader.channel_info.items():
                if info_ch.get('marker'):
                    marker_list.append({
                        'Canal': ch,
                        'Marqueur': info_ch['marker'],
                        'Description': info_ch.get('pns', '')
                    })
            
            if marker_list:
                st.dataframe(pd.DataFrame(marker_list), use_container_width=True, height=200)
            else:
                st.warning("Aucun marqueur identifié automatiquement.")
        
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
                
                # Analyser les marqueurs
                channels_with_markers = reader.get_channels_with_markers()
                
                if channels_with_markers:
                    st.info(f"🔍 Analyse de {len(channels_with_markers)} marqueurs...")
                    
                    parent_gate = gates.get('viable', gates.get('singlets', pd.Series(True, index=data.index)))
                    
                    progress_bar = st.progress(0)
                    
                    for i, channel in enumerate(channels_with_markers):
                        try:
                            gate, pct, stats = gate_positive_simple(data, channel, parent_gate)
                            
                            if stats and stats.get('count', 0) > 0:
                                marker = reader.get_marker_for_channel(channel)
                                gates[f'{marker}_pos'] = gate
                                stats['parent'] = 'viable' if 'viable' in gates else 'singlets'
                                stats['marker'] = marker
                                marker_stats[channel] = stats
                        except Exception as e:
                            pass
                        
                        progress_bar.progress((i + 1) / len(channels_with_markers))
                    
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
                "🎨 Visualisations", 
                "🔬 Marqueurs",
                "💾 Export"
            ])
            
            # TAB 1 : STATISTIQUES
            with tab1:
                st.markdown("#### Populations Principales")
                
                stats_data = []
                total = len(reader.data)
                
                for gate_name, mask in st.session_state.gates.items():
                    if '_pos' not in gate_name:
                        stats_data.append({
                            'Population': gate_name,
                            'Nombre': int(mask.sum()),
                            '% Total': f"{(mask.sum() / total) * 100:.1f}%"
                        })
                
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            # TAB 2 : VISUALISATIONS
            with tab2:
                st.markdown("#### 🎨 Grille de Visualisations avec Marqueurs")
                
                all_channels = reader.channels
                
                # Canaux scatter
                fsc_a = [c for c in all_channels if 'FSC' in c.upper() and '-A' in c.upper()]
                ssc_a = [c for c in all_channels if 'SSC' in c.upper() and '-A' in c.upper()]
                fsc_h = [c for c in all_channels if 'FSC' in c.upper() and '-H' in c.upper()]
                
                # Paires par défaut
                default_pairs = []
                if fsc_a and ssc_a:
                    default_pairs.append((fsc_a[0], ssc_a[0]))
                if fsc_a and fsc_h:
                    default_pairs.append((fsc_a[0], fsc_h[0]))
                
                # Ajouter les canaux avec marqueurs
                marker_channels = reader.get_channels_with_markers()
                for ch in marker_channels[:6]:
                    if ssc_a:
                        default_pairs.append((ch, ssc_a[0]))
                
                # Contrôles
                col1, col2 = st.columns(2)
                with col1:
                    n_plots = st.slider("Nombre de graphiques", 1, 12, min(6, len(default_pairs)))
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    x_min = st.number_input("X min", value=0.0, step=0.5)
                with col2:
                    x_max = st.number_input("X max", value=6.0, step=0.5)
                with col3:
                    y_min = st.number_input("Y min", value=0.0, step=0.5)
                with col4:
                    y_max = st.number_input("Y max", value=6.0, step=0.5)
                
                use_custom_limits = st.checkbox("Appliquer limites personnalisées", value=False)
                
                if st.button("📊 Générer la grille", type="primary"):
                    with st.spinner("Génération..."):
                        pairs_to_plot = default_pairs[:n_plots]
                        
                        xlims = [(x_min, x_max)] * n_plots if use_custom_limits else None
                        ylims = [(y_min, y_max)] * n_plots if use_custom_limits else None
                        
                        display_gates = {k: v for k, v in st.session_state.gates.items() 
                                        if k in ['singlets', 'viable']}
                        
                        fig = create_multi_plot_figure(
                            reader.data, pairs_to_plot, display_gates, reader,
                            log_scale=log_scale, xlims=xlims, ylims=ylims, ncols=ncols
                        )
                        
                        if fig:
                            st.pyplot(fig)
                            st.session_state.grid_figure = fig
                
                if 'grid_figure' in st.session_state and st.session_state.grid_figure:
                    buf = io.BytesIO()
                    st.session_state.grid_figure.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        "📥 Télécharger Grille (PNG 300 DPI)",
                        buf,
                        f"grille_{reader.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        "image/png",
                        use_container_width=True
                    )
            
            # TAB 3 : MARQUEURS DÉTAILLÉS
            with tab3:
                st.markdown("#### 🔬 Statistiques par Marqueur")
                
                if st.session_state.marker_stats:
                    marker_df = pd.DataFrame([
                        {
                            'Marqueur': reader.get_marker_for_channel(channel) or channel,
                            'Canal': channel,
                            '% Positif': f"{stats['percentage']}%",
                            'Count': stats['count'],
                            'MFI Mean': stats['mean'],
                            'MFI Median': stats['median']
                        }
                        for channel, stats in st.session_state.marker_stats.items()
                    ])
                    
                    # Trier par pourcentage
                    marker_df['pct_sort'] = marker_df['% Positif'].str.replace('%', '').astype(float)
                    marker_df = marker_df.sort_values('pct_sort', ascending=False).drop('pct_sort', axis=1)
                    
                    st.dataframe(marker_df, use_container_width=True, height=400)
                    
                    # Graphique
                    st.markdown("##### Distribution des % Positifs par Marqueur")
                    
                    fig, ax = plt.subplots(figsize=(12, max(6, len(marker_df) * 0.4)))
                    
                    markers = marker_df['Marqueur'].tolist()
                    percentages = [float(p.replace('%', '')) for p in marker_df['% Positif'].tolist()]
                    
                    colors = plt.cm.RdYlGn(np.array(percentages) / 100)
                    
                    bars = ax.barh(range(len(markers)), percentages, color=colors)
                    ax.set_yticks(range(len(markers)))
                    ax.set_yticklabels(markers, fontsize=10)
                    ax.set_xlabel('% Positif', fontsize=12)
                    ax.set_title('Pourcentage de Cellules Positives par Marqueur', fontsize=14, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    ax.invert_yaxis()
                    
                    # Ajouter les valeurs sur les barres
                    for i, (bar, pct) in enumerate(zip(bars, percentages)):
                        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                               f'{pct:.1f}%', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # TAB 4 : EXPORT
            with tab4:
                st.markdown("#### 💾 Export Complet")
                
                st.info("📦 L'export Excel inclut :\n"
                       "- **Mapping Canaux → Marqueurs** (CD4, CD8, CD3, etc.)\n"
                       "- Statistiques par marqueur (%, MFI)\n"
                       "- Tableau format publication\n"
                       "- Données avec noms de marqueurs")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Générer Excel Complet", type="primary", use_container_width=True):
                        with st.spinner("Génération..."):
                            excel_data = export_complete_excel(
                                reader, 
                                st.session_state.gates, 
                                st.session_state.marker_stats,
                                reader.filename
                            )
                            
                            if excel_data:
                                st.download_button(
                                    "📥 Télécharger Excel",
                                    excel_data,
                                    f"FACS_{reader.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                
                with col2:
                    if st.session_state.marker_stats:
                        csv_data = pd.DataFrame([
                            {
                                'Echantillon': reader.filename,
                                'Marqueur': reader.get_marker_for_channel(channel) or channel,
                                'Canal': channel,
                                'Percentage': stats['percentage'],
                                'Count': stats['count'],
                                'MFI_Mean': stats['mean'],
                                'MFI_Median': stats['median']
                            }
                            for channel, stats in st.session_state.marker_stats.items()
                        ])
                        
                        st.download_button(
                            "📋 Télécharger CSV Marqueurs",
                            csv_data.to_csv(index=False),
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
    <p>🔬 <b>FACS Autogating - Identification des Marqueurs</b></p>
    <p>CD4 • CD8 • CD3 • FoxP3 • et plus | Grille de visualisations | Export Excel détaillé</p>
</div>
""", unsafe_allow_html=True)
