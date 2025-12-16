#!/usr/bin/env python3
"""
Application Streamlit pour l'Automatisation du Gating FACS
Interface utilisateur conviviale pour analyse de cytométrie en flux
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

# Import des modules du pipeline
from facs_autogating import FCSGatingPipeline
from facs_workflows_advanced import BatchFCSAnalysis, AdvancedGatingStrategies
from facs_utilities import FCSValidator, ChannelDetector, StatisticsExporter

# Configuration de la page
st.set_page_config(
    page_title="FACS Autogating Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
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
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown('<h1 class="main-header">🔬 Pipeline d\'Automatisation du Gating FACS</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Navigation
with st.sidebar:
    st.image("https://via.placeholder.com/200x100.png?text=FACS+Pipeline", use_container_width=True)
    st.markdown("## 📋 Navigation")
    
    mode = st.radio(
        "Choisir le mode d'analyse :",
        ["🔍 Analyse Simple", "📊 Analyse par Lot", "✅ Validation de Fichiers", "🎯 Détection Automatique"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres Globaux")
    
    apply_compensation = st.checkbox("Appliquer compensation spectrale", value=True)
    transform_type = st.selectbox(
        "Type de transformation",
        ["logicle", "asinh", "hyperlog", "aucune"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📚 Ressources")
    st.markdown("""
    - [Documentation](https://github.com/votre-repo)
    - [Tutoriel](https://github.com/votre-repo/tutorial)
    - [Références](https://github.com/votre-repo/references)
    """)

# Session state pour stocker les données
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# ========================================
# MODE 1: ANALYSE SIMPLE
# ========================================
if mode == "🔍 Analyse Simple":
    st.markdown('<h2 class="section-header">🔍 Analyse d\'un Fichier FCS</h2>', unsafe_allow_html=True)
    
    # Upload de fichier
    uploaded_file = st.file_uploader(
        "📁 Télécharger un fichier FCS",
        type=['fcs'],
        help="Sélectionnez un fichier FCS à analyser"
    )
    
    if uploaded_file is not None:
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        st.markdown(f'<div class="info-box">📄 Fichier chargé : <b>{uploaded_file.name}</b></div>', unsafe_allow_html=True)
        
        # Validation rapide
        with st.spinner("Validation du fichier..."):
            validation = FCSValidator.validate_fcs_file(tmp_path)
        
        if not validation['is_valid']:
            st.error(f"❌ Fichier invalide : {validation['errors']}")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Événements", f"{validation['info']['event_count']:,}")
            with col2:
                st.metric("Canaux", validation['info']['channel_count'])
            with col3:
                st.metric("Taille", f"{validation['info']['file_size_mb']:.1f} MB")
            
            if validation['warnings']:
                with st.expander("⚠️ Avertissements"):
                    for warning in validation['warnings']:
                        st.warning(warning)
            
            # Configuration de l'analyse
            st.markdown("### ⚙️ Configuration du Gating")
            
            col1, col2 = st.columns(2)
            
            with col1:
                gate_singlets = st.checkbox("Gate Singlets (FSC-A vs FSC-H)", value=True)
                gate_debris = st.checkbox("Supprimer les débris (FSC/SSC)", value=True)
                
            with col2:
                gate_markers = st.checkbox("Gating automatique des marqueurs", value=True)
                create_quadrants = st.checkbox("Créer quadrants CD4/CD8", value=False)
            
            # Bouton d'analyse
            if st.button("🚀 Lancer l'Analyse", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours... ⏳"):
                    try:
                        # Chargement du pipeline
                        pipeline = FCSGatingPipeline(
                            tmp_path,
                            compensate=apply_compensation,
                            transform=transform_type if transform_type != "aucune" else None
                        )
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Étape 1: Singlets
                        if gate_singlets:
                            status_text.text("Étape 1/4: Gating des singlets...")
                            progress_bar.progress(25)
                            pipeline.gate_singlets_fsc_ssc()
                        
                        # Étape 2: Débris
                        if gate_debris:
                            status_text.text("Étape 2/4: Suppression des débris...")
                            progress_bar.progress(50)
                            parent = 'singlets' if gate_singlets else None
                            pipeline.gate_debris_removal(parent_gate=parent)
                        
                        # Étape 3: Marqueurs
                        if gate_markers:
                            status_text.text("Étape 3/4: Gating des marqueurs...")
                            progress_bar.progress(75)
                            
                            detected = ChannelDetector.detect_channels(pipeline)
                            
                            if 'T_CELLS' in detected:
                                cd3_channels = [ch for ch in detected['T_CELLS'] if 'CD3' in ch.upper()]
                                if cd3_channels:
                                    parent_gate = 'singlets_viable' if gate_debris else 'singlets' if gate_singlets else None
                                    pipeline.gate_gmm_1d(
                                        cd3_channels[0],
                                        n_components=2,
                                        select_component='positive',
                                        parent_gate=parent_gate
                                    )
                        
                        # Étape 4: Statistiques
                        status_text.text("Étape 4/4: Calcul des statistiques...")
                        progress_bar.progress(100)
                        stats = pipeline.compute_statistics()
                        
                        # Stocker dans session state
                        st.session_state.pipeline = pipeline
                        st.session_state.stats = stats
                        st.session_state.analysis_complete = True
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.markdown('<div class="success-box">✅ <b>Analyse terminée avec succès !</b></div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse : {str(e)}")
                        st.exception(e)
        
        # Affichage des résultats
        if st.session_state.analysis_complete and st.session_state.pipeline is not None:
            st.markdown("---")
            st.markdown('<h2 class="section-header">📊 Résultats de l\'Analyse</h2>', unsafe_allow_html=True)
            
            # Onglets pour organiser les résultats
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistiques", "🎨 Visualisations", "📋 Détails", "💾 Export"])
            
            with tab1:
                st.markdown("### 📊 Statistiques des Populations")
                
                # Résumé
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Populations identifiées", len(st.session_state.pipeline.gates))
                with col2:
                    st.metric("Événements totaux", f"{len(st.session_state.pipeline.data):,}")
                
                # Tableau des statistiques
                display_stats = st.session_state.stats[['Population', 'Count', 'Percentage_of_total']].copy()
                display_stats.columns = ['Population', 'Nombre', '% du Total']
                st.dataframe(display_stats, use_container_width=True, height=400)
                
                # Graphique en barres
                fig, ax = plt.subplots(figsize=(10, 6))
                top_pops = st.session_state.stats.nlargest(10, 'Count')
                ax.barh(top_pops['Population'], top_pops['Percentage_of_total'], color='steelblue')
                ax.set_xlabel('Pourcentage du Total (%)', fontsize=12)
                ax.set_title('Top 10 des Populations', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
                st.markdown("### 🎨 Visualisations des Gates")
                
                # Sélection des canaux
                available_channels = st.session_state.pipeline.channels
                
                col1, col2 = st.columns(2)
                with col1:
                    channel_x = st.selectbox("Canal X", available_channels, index=0)
                with col2:
                    channel_y = st.selectbox("Canal Y", available_channels, 
                                            index=1 if len(available_channels) > 1 else 0)
                
                # Sélection des gates à afficher
                available_gates = list(st.session_state.pipeline.gates.keys())
                selected_gates = st.multiselect(
                    "Sélectionner les populations à afficher",
                    available_gates,
                    default=available_gates[:3] if len(available_gates) >= 3 else available_gates
                )
                
                if selected_gates:
                    # Créer la visualisation
                    n_gates = len(selected_gates)
                    n_cols = min(2, n_gates)
                    n_rows = (n_gates + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
                    if n_gates == 1:
                        axes = [axes]
                    else:
                        axes = axes.flatten()
                    
                    for idx, gate_name in enumerate(selected_gates):
                        ax = axes[idx]
                        mask = st.session_state.pipeline.gates[gate_name]
                        
                        # Background
                        ax.scatter(
                            st.session_state.pipeline.data[channel_x],
                            st.session_state.pipeline.data[channel_y],
                            s=1, c='lightgray', alpha=0.3, rasterized=True
                        )
                        
                        # Foreground
                        ax.scatter(
                            st.session_state.pipeline.data.loc[mask, channel_x],
                            st.session_state.pipeline.data.loc[mask, channel_y],
                            s=1, c='red', alpha=0.6, rasterized=True
                        )
                        
                        ax.set_xlabel(channel_x, fontsize=10)
                        ax.set_ylabel(channel_y, fontsize=10)
                        pct = (mask.sum() / len(mask)) * 100
                        ax.set_title(f"{gate_name}\n{mask.sum():,} events ({pct:.1f}%)", fontsize=10)
                        ax.grid(True, alpha=0.3)
                    
                    # Masquer les axes non utilisés
                    for idx in range(n_gates, len(axes)):
                        axes[idx].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Sélectionnez au moins une population à visualiser")
            
            with tab3:
                st.markdown("### 📋 Informations Détaillées")
                
                # Informations du fichier
                with st.expander("📄 Informations du Fichier", expanded=True):
                    info_df = pd.DataFrame({
                        'Paramètre': [
                            'Fichier',
                            'Événements totaux',
                            'Nombre de canaux',
                            'Compensation',
                            'Transformation',
                            'Populations identifiées'
                        ],
                        'Valeur': [
                            uploaded_file.name,
                            f"{len(st.session_state.pipeline.data):,}",
                            len(st.session_state.pipeline.channels),
                            'Oui' if apply_compensation else 'Non',
                            transform_type if transform_type != "aucune" else 'Aucune',
                            len(st.session_state.pipeline.gates)
                        ]
                    })
                    st.table(info_df)
                
                # Liste des canaux
                with st.expander("📋 Canaux Disponibles"):
                    channels_df = pd.DataFrame({
                        'Index': range(1, len(st.session_state.pipeline.channels) + 1),
                        'Canal': st.session_state.pipeline.channels
                    })
                    st.dataframe(channels_df, use_container_width=True, height=300)
                
                # Détection automatique
                with st.expander("🔍 Marqueurs Détectés"):
                    detected = ChannelDetector.detect_channels(st.session_state.pipeline)
                    if detected:
                        for category, channels in detected.items():
                            st.markdown(f"**{category}:**")
                            for ch in channels:
                                st.markdown(f"- {ch}")
                    else:
                        st.info("Aucun marqueur standard détecté")
            
            with tab4:
                st.markdown("### 💾 Export des Résultats")
                
                # Export Excel
                st.markdown("#### 📊 Fichier Excel Complet")
                
                if st.button("Générer fichier Excel", type="primary"):
                    with st.spinner("Génération du fichier Excel..."):
                        # Créer un buffer
                        buffer = io.BytesIO()
                        
                        # Export temporaire
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
                            st.session_state.pipeline.export_to_excel(
                                tmp_excel.name,
                                include_populations=True
                            )
                            
                            # Lire le fichier
                            with open(tmp_excel.name, 'rb') as f:
                                buffer.write(f.read())
                        
                        buffer.seek(0)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"FACS_analysis_{timestamp}.xlsx"
                        
                        st.download_button(
                            label="📥 Télécharger Excel",
                            data=buffer,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success("✅ Fichier Excel prêt au téléchargement !")
                
                # Export CSV des statistiques
                st.markdown("#### 📋 Statistiques (CSV)")
                csv = st.session_state.stats.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv,
                    file_name=f"FACS_stats_{timestamp}.csv",
                    mime="text/csv"
                )

# ========================================
# MODE 2: ANALYSE PAR LOT
# ========================================
elif mode == "📊 Analyse par Lot":
    st.markdown('<h2 class="section-header">📊 Analyse par Lot de Fichiers FCS</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Téléchargez plusieurs fichiers FCS pour une analyse comparative</div>', unsafe_allow_html=True)
    
    # Upload multiple
    uploaded_files = st.file_uploader(
        "📁 Télécharger plusieurs fichiers FCS",
        type=['fcs'],
        accept_multiple_files=True,
        help="Sélectionnez tous les fichiers FCS à analyser en lot"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} fichiers chargés")
        
        # Afficher la liste
        with st.expander("📋 Fichiers chargés", expanded=True):
            file_list = pd.DataFrame({
                'Fichier': [f.name for f in uploaded_files],
                'Taille (KB)': [f.size / 1024 for f in uploaded_files]
            })
            st.dataframe(file_list, use_container_width=True)
        
        # Configuration
        st.markdown("### ⚙️ Configuration de l'Analyse")
        
        col1, col2 = st.columns(2)
        with col1:
            gate_strategy = st.selectbox(
                "Stratégie de gating",
                ["standard", "lymphocytes"],
                help="Standard : QC basique. Lymphocytes : Panel T cells complet"
            )
        
        with col2:
            include_comparisons = st.checkbox("Générer graphiques comparatifs", value=True)
        
        # Bouton d'analyse
        if st.button("🚀 Lancer l'Analyse par Lot", type="primary", use_container_width=True):
            with st.spinner("Analyse des fichiers en cours... ⏳"):
                try:
                    # Sauvegarder temporairement
                    temp_files = []
                    sample_names = []
                    
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp:
                            tmp.write(uploaded_file.read())
                            temp_files.append(tmp.name)
                            sample_names.append(Path(uploaded_file.name).stem)
                    
                    # Analyse batch
                    batch = BatchFCSAnalysis(temp_files, sample_names)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Traitement des fichiers...")
                    pipelines = batch.run_standard_pipeline(
                        compensate=apply_compensation,
                        transform=transform_type if transform_type != "aucune" else None,
                        gate_strategy=gate_strategy
                    )
                    progress_bar.progress(50)
                    
                    status_text.text("Génération des comparaisons...")
                    comparison = batch.compare_populations()
                    progress_bar.progress(100)
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Stocker les résultats
                    st.session_state.batch = batch
                    st.session_state.comparison = comparison
                    st.session_state.batch_complete = True
                    
                    st.markdown('<div class="success-box">✅ <b>Analyse par lot terminée !</b></div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
                    st.exception(e)
        
        # Affichage des résultats
        if 'batch_complete' in st.session_state and st.session_state.batch_complete:
            st.markdown("---")
            st.markdown('<h2 class="section-header">📊 Résultats Comparatifs</h2>', unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["📈 Vue d'ensemble", "🎨 Graphiques", "💾 Export"])
            
            with tab1:
                st.markdown("### 📊 Tableau Comparatif")
                
                if st.session_state.comparison is not None:
                    # Afficher les comptages
                    st.markdown("#### Comptages des Populations")
                    counts_df = st.session_state.comparison['Count']
                    st.dataframe(counts_df, use_container_width=True, height=400)
                    
                    # Afficher les pourcentages
                    st.markdown("#### Pourcentages du Total")
                    pct_df = st.session_state.comparison['Percentage_of_total']
                    st.dataframe(pct_df.style.format("{:.2f}%"), use_container_width=True, height=400)
            
            with tab2:
                st.markdown("### 🎨 Graphiques Comparatifs")
                
                if include_comparisons and st.session_state.batch.comparative_stats is not None:
                    # Sélection des populations à comparer
                    all_populations = st.session_state.batch.comparative_stats['Population'].unique()
                    selected_pops = st.multiselect(
                        "Sélectionner les populations à comparer",
                        all_populations,
                        default=list(all_populations[:5])
                    )
                    
                    if selected_pops:
                        plot_data = st.session_state.batch.comparative_stats[
                            st.session_state.batch.comparative_stats['Population'].isin(selected_pops)
                        ]
                        
                        # Graphique en barres groupées
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        pivot_data = plot_data.pivot(
                            index='Sample',
                            columns='Population',
                            values='Percentage_of_total'
                        )
                        
                        pivot_data.plot(kind='bar', ax=ax, width=0.8)
                        ax.set_xlabel('Échantillon', fontsize=12)
                        ax.set_ylabel('Pourcentage (%)', fontsize=12)
                        ax.set_title('Comparaison des Populations', fontsize=14, fontweight='bold')
                        ax.legend(title='Population', bbox_to_anchor=(1.05, 1), loc='upper left')
                        ax.grid(axis='y', alpha=0.3)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
            
            with tab3:
                st.markdown("### 💾 Export des Résultats")
                
                if st.button("Générer Export Comparatif", type="primary"):
                    with st.spinner("Génération du fichier..."):
                        buffer = io.BytesIO()
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                            st.session_state.batch.export_comparative_excel(tmp.name)
                            
                            with open(tmp.name, 'rb') as f:
                                buffer.write(f.read())
                        
                        buffer.seek(0)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        st.download_button(
                            label="📥 Télécharger Excel Comparatif",
                            data=buffer,
                            file_name=f"FACS_batch_comparison_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

# ========================================
# MODE 3: VALIDATION
# ========================================
elif mode == "✅ Validation de Fichiers":
    st.markdown('<h2 class="section-header">✅ Validation de Fichiers FCS</h2>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📁 Télécharger fichiers FCS à valider",
        type=['fcs'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🔍 Valider les Fichiers", type="primary"):
        with st.spinner("Validation en cours..."):
            results = []
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp:
                    tmp.write(uploaded_file.read())
                    validation = FCSValidator.validate_fcs_file(tmp.name)
                    
                    results.append({
                        'Fichier': uploaded_file.name,
                        'Valide': '✅' if validation['is_valid'] else '❌',
                        'Événements': validation['info'].get('event_count', 0),
                        'Canaux': validation['info'].get('channel_count', 0),
                        'Compensation': '✅' if validation['info'].get('has_compensation', False) else '❌',
                        'Taille (MB)': f"{validation['info'].get('file_size_mb', 0):.1f}",
                        'Avertissements': '; '.join(validation['warnings']) if validation['warnings'] else 'Aucun'
                    })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Résumé
            n_valid = (results_df['Valide'] == '✅').sum()
            n_total = len(results_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fichiers valides", f"{n_valid}/{n_total}")
            with col2:
                st.metric("Fichiers invalides", f"{n_total - n_valid}/{n_total}")
            with col3:
                avg_events = results_df['Événements'].mean()
                st.metric("Événements moyens", f"{avg_events:,.0f}")

# ========================================
# MODE 4: DÉTECTION AUTOMATIQUE
# ========================================
elif mode == "🎯 Détection Automatique":
    st.markdown('<h2 class="section-header">🎯 Détection Automatique de Marqueurs</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Analysez votre fichier FCS pour détecter automatiquement les marqueurs et obtenir une suggestion de workflow</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📁 Télécharger un fichier FCS",
        type=['fcs']
    )
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        if st.button("🔍 Analyser et Suggérer Workflow", type="primary"):
            with st.spinner("Analyse en cours..."):
                try:
                    # Charger le pipeline
                    pipeline = FCSGatingPipeline(tmp_path, compensate=False, transform=None)
                    
                    # Détection
                    detected = ChannelDetector.detect_channels(pipeline)
                    suggestions = ChannelDetector.suggest_gating_strategy(detected)
                    
                    # Affichage des résultats
                    st.markdown("### 🔬 Marqueurs Détectés")
                    
                    if detected:
                        cols = st.columns(2)
                        for idx, (category, channels) in enumerate(detected.items()):
                            with cols[idx % 2]:
                                with st.expander(f"**{category}**", expanded=True):
                                    for ch in channels:
                                        st.markdown(f"- {ch}")
                    else:
                        st.info("Aucun marqueur standard détecté")
                    
                    # Suggestions
                    st.markdown("### 💡 Workflow Suggéré")
                    
                    if suggestions:
                        for suggestion in suggestions:
                            st.markdown(f"- {suggestion}")
                    else:
                        st.info("Workflow standard recommandé")
                    
                    # Code généré
                    st.markdown("### 💻 Code Python Généré")
                    
                    from facs_utilities import auto_suggest_workflow
                    code = auto_suggest_workflow(tmp_path)
                    
                    st.code(code, language='python')
                    
                    # Download button
                    st.download_button(
                        label="📥 Télécharger le Code",
                        data=code,
                        file_name="suggested_workflow.py",
                        mime="text/x-python"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p>🔬 <b>FACS Autogating Pipeline</b> | Version 1.0</p>
    <p>Développé avec ❤️ pour la recherche en biologie</p>
    <p><a href="https://github.com/votre-repo">GitHub</a> | <a href="https://github.com/votre-repo/docs">Documentation</a></p>
</div>
""", unsafe_allow_html=True)
