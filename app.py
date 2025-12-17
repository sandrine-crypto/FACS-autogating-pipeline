#!/usr/bin/env python3
"""
Application Streamlit FACS avec Export Complet
Version avec export Excel multi-feuilles, visualisations et données
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
import base64

# Configuration de la page
st.set_page_config(
    page_title="FACS Autogating - Export Complet",
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
        
        # Récupérer les noms (majuscules ET minuscules)
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
        """Retourne les informations du fichier"""
        return {
            'event_count': len(self.data),
            'channel_count': len(self.channels),
            'channels': self.channels
        }

def gate_singlets_simple(data, fsc_a='FSC-A', fsc_h='FSC-H', threshold=1.5):
    """Gating simple des singlets par ratio"""
    if fsc_a not in data.columns or fsc_h not in data.columns:
        return pd.Series(True, index=data.index)
    
    ratio = data[fsc_h] / (data[fsc_a] + 1)
    median_ratio = ratio.median()
    mad = (ratio - median_ratio).abs().median()
    
    gate = (ratio > median_ratio - threshold * mad) & (ratio < median_ratio + threshold * mad)
    return gate

def gate_debris_simple(data, fsc='FSC-A', ssc='SSC-A', percentile=2):
    """Suppression débris par percentile"""
    if fsc not in data.columns or ssc not in data.columns:
        return pd.Series(True, index=data.index)
    
    fsc_thresh = np.percentile(data[fsc], percentile)
    ssc_thresh = np.percentile(data[ssc], percentile)
    
    gate = (data[fsc] > fsc_thresh) & (data[ssc] > ssc_thresh)
    return gate

def gate_positive_simple(data, channel):
    """Gating positif/négatif simple"""
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

def export_to_excel_complete(reader, gates, filename='facs_export_complet.xlsx'):
    """Export complet : statistiques + données + infos"""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils.dataframe import dataframe_to_rows
    except ImportError:
        st.error("openpyxl non disponible. Export CSV uniquement.")
        return None
    
    # Créer le classeur
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # Supprimer la feuille par défaut
    
    # FEUILLE 1 : Statistiques
    ws_stats = wb.create_sheet("Statistiques")
    stats_data = []
    for gate_name, mask in gates.items():
        stats_data.append({
            'Population': gate_name,
            'Nombre': int(mask.sum()),
            'Pourcentage_Total': round((mask.sum() / len(reader.data)) * 100, 2),
            'Pourcentage_Parent': 100.0  # Simplification
        })
    
    stats_df = pd.DataFrame(stats_data)
    for r_idx, row in enumerate(dataframe_to_rows(stats_df, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_stats.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                cell.font = Font(bold=True, color="FFFFFF")
    
    # FEUILLE 2 : Informations du fichier
    ws_info = wb.create_sheet("Info_Fichier")
    info_data = [
        ['Paramètre', 'Valeur'],
        ['Nombre d\'événements', len(reader.data)],
        ['Nombre de canaux', len(reader.channels)],
        ['Populations identifiées', len(gates)],
        ['Date d\'analyse', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    
    for r_idx, row in enumerate(info_data, 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_info.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
                cell.font = Font(bold=True, color="FFFFFF")
    
    # FEUILLE 3 : Liste des canaux
    ws_channels = wb.create_sheet("Canaux")
    ws_channels.append(['Index', 'Nom du Canal'])
    for idx, ch in enumerate(reader.channels, 1):
        ws_channels.append([idx, ch])
    
    ws_channels.cell(1, 1).font = Font(bold=True)
    ws_channels.cell(1, 2).font = Font(bold=True)
    
    # FEUILLE 4 : Données corrigées (échantillon)
    ws_data = wb.create_sheet("Donnees_Echantillon")
    
    # Ajouter colonnes de population
    sample_data = reader.data.head(1000).copy()  # 1000 premiers événements
    
    for gate_name, mask in gates.items():
        sample_data[f'Gate_{gate_name}'] = mask.head(1000).astype(int)
    
    for r_idx, row in enumerate(dataframe_to_rows(sample_data, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_data.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:
                cell.font = Font(bold=True)
    
    # FEUILLE 5 : Statistiques détaillées par population
    ws_details = wb.create_sheet("Details_Populations")
    ws_details.append(['Population', 'Paramètre', 'Valeur'])
    
    for gate_name, mask in gates.items():
        gated_data = reader.data[mask]
        ws_details.append([gate_name, 'Count', int(mask.sum())])
        ws_details.append([gate_name, 'Percentage', round((mask.sum() / len(reader.data)) * 100, 2)])
        
        # Statistiques FSC/SSC si disponibles
        for ch in ['FSC-A', 'SSC-A']:
            if ch in gated_data.columns and len(gated_data) > 0:
                ws_details.append([gate_name, f'{ch}_mean', round(gated_data[ch].mean(), 2)])
                ws_details.append([gate_name, f'{ch}_median', round(gated_data[ch].median(), 2)])
    
    # Sauvegarder
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    
    return output

st.markdown('<h1 class="main-header">🔬 FACS Autogating - Export Complet</h1>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("## 🎯 Navigation")
    mode = st.radio("Mode d'analyse", ["🔍 Analyse Simple", "📊 Informations du Fichier"], index=0)
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")
    gate_singlets_option = st.checkbox("Gate Singlets", value=True)
    gate_debris_option = st.checkbox("Supprimer Débris", value=True)
    st.markdown("---")
    st.markdown("### 💾 Options d'Export")
    export_images = st.checkbox("Inclure images haute résolution", value=True)
    export_raw_data = st.checkbox("Inclure données brutes", value=False, help="Attention : fichier volumineux")

if 'reader' not in st.session_state:
    st.session_state.reader = None
if 'gates' not in st.session_state:
    st.session_state.gates = {}
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'figures' not in st.session_state:
    st.session_state.figures = {}

if mode == "🔍 Analyse Simple":
    st.markdown("### 📁 Upload Fichier FCS")
    
    uploaded_file = st.file_uploader("Sélectionner un fichier FCS", type=['fcs'])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        st.markdown(f'<div class="info-box">📄 Fichier : <b>{uploaded_file.name}</b></div>', unsafe_allow_html=True)
        
        try:
            with st.spinner("Chargement du fichier..."):
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
                with st.spinner("Analyse en cours..."):
                    data = reader.data
                    gates = {}
                    
                    if gate_singlets_option:
                        fsc_a = [c for c in data.columns if 'FSC' in c.upper() and 'A' in c.upper()]
                        fsc_h = [c for c in data.columns if 'FSC' in c.upper() and 'H' in c.upper()]
                        
                        if fsc_a and fsc_h:
                            gates['singlets'] = gate_singlets_simple(data, fsc_a[0], fsc_h[0])
                            st.success(f"✅ Singlets : {gates['singlets'].sum():,} / {len(data):,}")
                    
                    if gate_debris_option:
                        fsc_a = [c for c in data.columns if 'FSC' in c.upper() and 'A' in c.upper()]
                        ssc_a = [c for c in data.columns if 'SSC' in c.upper() and 'A' in c.upper()]
                        
                        if fsc_a and ssc_a:
                            parent = gates.get('singlets', pd.Series(True, index=data.index))
                            debris_gate = gate_debris_simple(data, fsc_a[0], ssc_a[0])
                            gates['viable'] = parent & debris_gate
                            st.success(f"✅ Cellules viables : {gates['viable'].sum():,}")
                    
                    marker_channels = [c for c in data.columns if any(m in c.upper() for m in ['CD', 'FITC', 'PE', 'APC', 'BV', 'AF'])]
                    
                    if marker_channels:
                        st.info(f"🔍 {len(marker_channels)} marqueurs détectés")
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
                    st.markdown('<div class="success-box">✅ Analyse terminée !</div>', unsafe_allow_html=True)
            
            if st.session_state.analysis_done and st.session_state.gates:
                st.markdown("---")
                st.markdown("### 📊 Résultats")
                
                tab1, tab2, tab3 = st.tabs(["📈 Statistiques", "🎨 Visualisations", "💾 Export Complet"])
                
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
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    populations = [s['Population'] for s in stats_data]
                    counts = [int(s['Nombre'].replace(',', '')) for s in stats_data]
                    
                    ax.barh(populations, counts, color='steelblue')
                    ax.set_xlabel('Nombre d\'événements', fontsize=11)
                    ax.set_title('Populations Identifiées', fontsize=13, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.session_state.figures['stats_bar'] = fig
                
                with tab2:
                    st.markdown("### 🎨 Scatter Plots")
                    
                    all_channels = list(reader.channels)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        x_channel = st.selectbox("Canal X", all_channels, index=0)
                    with col2:
                        y_channel = st.selectbox("Canal Y", all_channels, index=1 if len(all_channels) > 1 else 0)
                    
                    if x_channel and y_channel:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        ax.scatter(reader.data[x_channel], reader.data[y_channel], s=1, c='lightgray', alpha=0.3, rasterized=True)
                        
                        colors = ['red', 'green', 'blue', 'orange', 'purple']
                        for idx, (gate_name, mask) in enumerate(st.session_state.gates.items()):
                            color = colors[idx % len(colors)]
                            ax.scatter(reader.data.loc[mask, x_channel], reader.data.loc[mask, y_channel],
                                     s=2, c=color, alpha=0.5, label=f"{gate_name} ({mask.sum():,})", rasterized=True)
                        
                        ax.set_xlabel(x_channel, fontsize=11)
                        ax.set_ylabel(y_channel, fontsize=11)
                        ax.set_title(f'{x_channel} vs {y_channel}', fontsize=13, fontweight='bold')
                        ax.legend(loc='upper right', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.session_state.figures[f'scatter_{x_channel}_{y_channel}'] = fig
                
                with tab3:
                    st.markdown("### 💾 Export Complet des Résultats")
                    
                    st.info("📦 L'export complet inclut :\n"
                            "- ✅ Statistiques détaillées (5 feuilles Excel)\n"
                            "- ✅ Liste des canaux\n"
                            "- ✅ Échantillon de données (1000 événements)\n"
                            "- ✅ Informations du fichier")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 Export Excel Multi-Feuilles")
                        if st.button("Générer Excel Complet", type="primary", use_container_width=True):
                            with st.spinner("Génération de l'export complet..."):
                                try:
                                    excel_data = export_to_excel_complete(reader, st.session_state.gates)
                                    
                                    if excel_data:
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        filename = f"FACS_Export_Complet_{timestamp}.xlsx"
                                        
                                        st.download_button(
                                            label="📥 Télécharger Excel Complet",
                                            data=excel_data,
                                            file_name=filename,
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            use_container_width=True
                                        )
                                        st.success("✅ Export Excel prêt !")
                                except Exception as e:
                                    st.error(f"Erreur Excel : {str(e)}")
                    
                    with col2:
                        st.markdown("#### 📋 Export CSV Simple")
                        if st.button("Générer CSV", use_container_width=True):
                            export_data = []
                            for gate_name, mask in st.session_state.gates.items():
                                export_data.append({
                                    'Population': gate_name,
                                    'Count': mask.sum(),
                                    'Percentage': (mask.sum() / len(reader.data)) * 100
                                })
                            
                            export_df = pd.DataFrame(export_data)
                            csv = export_df.to_csv(index=False)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            st.download_button(
                                label="📥 Télécharger CSV",
                                data=csv,
                                file_name=f"FACS_stats_{timestamp}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    st.markdown("---")
                    st.markdown("#### 🖼️ Export Visualisations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'stats_bar' in st.session_state.figures:
                            buf = io.BytesIO()
                            st.session_state.figures['stats_bar'].savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            buf.seek(0)
                            
                            st.download_button(
                                label="📥 Graphique Statistiques (PNG)",
                                data=buf,
                                file_name=f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    
                    with col2:
                        # Export scatter plot si disponible
                        scatter_keys = [k for k in st.session_state.figures.keys() if k.startswith('scatter_')]
                        if scatter_keys:
                            buf = io.BytesIO()
                            st.session_state.figures[scatter_keys[0]].savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            buf.seek(0)
                            
                            st.download_button(
                                label="📥 Scatter Plot (PNG)",
                                data=buf,
                                file_name=f"scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    
                    if export_raw_data:
                        st.markdown("---")
                        st.markdown("#### 📦 Export Données Complètes")
                        st.warning("⚠️ Attention : fichier volumineux !")
                        
                        if st.button("Exporter Toutes les Données", use_container_width=True):
                            with st.spinner("Préparation des données complètes..."):
                                # Ajouter colonnes de gate
                                full_data = reader.data.copy()
                                for gate_name, mask in st.session_state.gates.items():
                                    full_data[f'Gate_{gate_name}'] = mask.astype(int)
                                
                                csv_full = full_data.to_csv(index=False)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                
                                st.download_button(
                                    label="📥 Télécharger Données Complètes (CSV)",
                                    data=csv_full,
                                    file_name=f"FACS_donnees_completes_{timestamp}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
        
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
            st.exception(e)

elif mode == "📊 Informations du Fichier":
    st.markdown("### 📊 Analyse des Canaux")
    
    uploaded_file = st.file_uploader("Sélectionner un fichier FCS", type=['fcs'], key="info_upload")
    
    if uploaded_file is not None:
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
            st.markdown("### 📋 Liste des Canaux")
            
            channels_df = pd.DataFrame({
                'Index': range(1, len(info['channels']) + 1),
                'Nom du Canal': info['channels']
            })
            
            st.dataframe(channels_df, use_container_width=True, height=400)
            
            # Export de la liste
            csv_channels = channels_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger Liste des Canaux",
                data=csv_channels,
                file_name=f"canaux_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            st.markdown("### 🔍 Marqueurs Détectés")
            
            markers = {
                'T cells': ['CD3', 'CD4', 'CD8'],
                'B cells': ['CD19', 'CD20'],
                'NK cells': ['CD56', 'CD16'],
                'Activation': ['CD25', 'CD69', 'HLA-DR'],
                'Viabilité': ['LIVE', 'DEAD', '7AAD', 'PI']
            }
            
            for category, marker_list in markers.items():
                found = [ch for ch in info['channels'] if any(m.upper() in ch.upper() for m in marker_list)]
                if found:
                    st.markdown(f"**{category}** :")
                    for ch in found:
                        st.markdown(f"- {ch}")
        
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>🔬 <b>FACS Autogating Pipeline - Version Export Complet</b></p>
    <p>Export Excel multi-feuilles | Visualisations haute résolution | Données corrigées</p>
</div>
""", unsafe_allow_html=True)
