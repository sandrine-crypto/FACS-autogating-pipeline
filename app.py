#!/usr/bin/env python3
"""
Application Streamlit Simplifiée pour FACS - Compatible Streamlit Cloud
Version finale avec reshape correct des données FCS
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

# Configuration de la page
st.set_page_config(
    page_title="FACS Autogating - Demo",
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
        # Récupérer les données brutes
        events = self.flow_data.events
        n_channels = self.flow_data.channel_count
        
        # Convertir en numpy array
        if not isinstance(events, np.ndarray):
            events = np.array(events, dtype=np.float64)
        
        # Reshape en (n_events, n_channels) si nécessaire
        if events.ndim == 1:
            n_events = len(events) // n_channels
            events = events.reshape(n_events, n_channels)
        
        # Récupérer les noms de canaux (essayer majuscules ET minuscules)
        pnn_labels = []
        for i in range(1, n_channels + 1):
            # Essayer $P1N (majuscules)
            pnn = self.flow_data.text.get(f'$P{i}N', None)
            # Si pas trouvé, essayer p1n (minuscules)
            if pnn is None or pnn == f'Channel_{i}':
                pnn = self.flow_data.text.get(f'p{i}n', f'Channel_{i}')
            # Nettoyer les espaces
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

st.markdown('<h1 class="main-header">🔬 FACS Autogating - Version Demo</h1>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("## 🎯 Navigation")
    mode = st.radio("Mode d'analyse", ["🔍 Analyse Simple", "📊 Informations du Fichier"], index=0)
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")
    gate_singlets_option = st.checkbox("Gate Singlets", value=True)
    gate_debris_option = st.checkbox("Supprimer Débris", value=True)

if 'reader' not in st.session_state:
    st.session_state.reader = None
if 'gates' not in st.session_state:
    st.session_state.gates = {}
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

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
                        fsc_a = [c for c in data.columns if 'FSC' in c and 'A' in c]
                        fsc_h = [c for c in data.columns if 'FSC' in c and 'H' in c]
                        
                        if fsc_a and fsc_h:
                            gates['singlets'] = gate_singlets_simple(data, fsc_a[0], fsc_h[0])
                            st.success(f"✅ Singlets : {gates['singlets'].sum():,} / {len(data):,}")
                    
                    if gate_debris_option:
                        fsc_a = [c for c in data.columns if 'FSC' in c and 'A' in c]
                        ssc_a = [c for c in data.columns if 'SSC' in c and 'A' in c]
                        
                        if fsc_a and ssc_a:
                            parent = gates.get('singlets', pd.Series(True, index=data.index))
                            debris_gate = gate_debris_simple(data, fsc_a[0], ssc_a[0])
                            gates['viable'] = parent & debris_gate
                            st.success(f"✅ Cellules viables : {gates['viable'].sum():,}")
                    
                    marker_channels = [c for c in data.columns if any(m in c.upper() for m in ['CD', 'FITC', 'PE', 'APC'])]
                    
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
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    populations = [s['Population'] for s in stats_data]
                    counts = [int(s['Nombre'].replace(',', '')) for s in stats_data]
                    
                    ax.barh(populations, counts, color='steelblue')
                    ax.set_xlabel('Nombre d\'événements', fontsize=11)
                    ax.set_title('Populations Identifiées', fontsize=13, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with tab2:
                    st.markdown("### 🎨 Scatter Plots")
                    
                    all_channels = list(reader.channels)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        x_channel = st.selectbox("Canal X", all_channels, index=0 if all_channels else None)
                    with col2:
                        y_channel = st.selectbox("Canal Y", all_channels, index=1 if len(all_channels) > 1 else 0)
                    
                    if x_channel and y_channel:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        ax.scatter(reader.data[x_channel], reader.data[y_channel], s=1, c='lightgray', alpha=0.3, rasterized=True)
                        
                        colors = ['red', 'green', 'blue', 'orange']
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
                
                with tab3:
                    st.markdown("### 💾 Export des Données")
                    
                    if st.button("Générer CSV", type="primary"):
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
                            file_name=f"facs_results_{timestamp}.csv",
                            mime="text/csv"
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
    <p>🔬 <b>FACS Autogating Pipeline - Version Demo</b></p>
    <p>Version simplifiée pour Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
