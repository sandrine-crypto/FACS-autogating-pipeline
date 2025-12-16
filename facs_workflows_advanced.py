#!/usr/bin/env python3
"""
Workflows avancés pour l'analyse FACS automatisée
Batch processing, analyse comparative, et méthodes avancées
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from facs_autogating import FCSGatingPipeline
import warnings
warnings.filterwarnings('ignore')


class BatchFCSAnalysis:
    """
    Analyse par lot de multiples fichiers FCS
    Permet la comparaison entre échantillons
    """
    
    def __init__(self, fcs_files: List[str], sample_names: Optional[List[str]] = None):
        """
        Args:
            fcs_files: Liste des chemins vers fichiers FCS
            sample_names: Noms personnalisés (None = utiliser noms de fichiers)
        """
        self.fcs_files = fcs_files
        self.sample_names = sample_names if sample_names else [Path(f).stem for f in fcs_files]
        self.pipelines = {}
        self.comparative_stats = None
        
    def run_standard_pipeline(self, 
                             compensate: bool = True,
                             transform: str = 'logicle',
                             gate_strategy: str = 'standard') -> Dict[str, FCSGatingPipeline]:
        """
        Exécute le pipeline standard sur tous les fichiers
        
        Args:
            compensate: Appliquer compensation
            transform: Type de transformation
            gate_strategy: 'standard', 'lymphocytes', 'custom'
            
        Returns:
            Dictionnaire {nom_échantillon: pipeline}
        """
        print(f"\n{'='*60}")
        print(f"ANALYSE PAR LOT: {len(self.fcs_files)} fichiers")
        print(f"{'='*60}\n")
        
        for fcs_file, sample_name in zip(self.fcs_files, self.sample_names):
            print(f"\n▶ Traitement de: {sample_name}")
            print("-" * 60)
            
            try:
                pipeline = FCSGatingPipeline(fcs_file, compensate, transform)
                
                # Application de la stratégie de gating
                if gate_strategy == 'standard':
                    self._apply_standard_gates(pipeline)
                elif gate_strategy == 'lymphocytes':
                    self._apply_lymphocyte_gates(pipeline)
                
                pipeline.compute_statistics()
                self.pipelines[sample_name] = pipeline
                
                print(f"✅ {sample_name}: {len(pipeline.gates)} populations identifiées")
                
            except Exception as e:
                print(f"❌ Erreur avec {sample_name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"✅ Analyse par lot terminée: {len(self.pipelines)}/{len(self.fcs_files)} fichiers traités")
        print(f"{'='*60}\n")
        
        return self.pipelines
    
    def _apply_standard_gates(self, pipeline: FCSGatingPipeline):
        """Stratégie de gating standard"""
        # Singlets
        if 'FSC-A' in pipeline.channels and 'FSC-H' in pipeline.channels:
            pipeline.gate_singlets_fsc_ssc()
        
        # Debris removal
        if 'FSC-A' in pipeline.channels and 'SSC-A' in pipeline.channels:
            parent = 'singlets' if 'singlets' in pipeline.gates else None
            pipeline.gate_debris_removal(parent_gate=parent)
    
    def _apply_lymphocyte_gates(self, pipeline: FCSGatingPipeline):
        """Stratégie spécifique pour analyse des lymphocytes"""
        # Singlets
        if 'FSC-A' in pipeline.channels and 'FSC-H' in pipeline.channels:
            pipeline.gate_singlets_fsc_ssc()
        
        # Lymphocytes par FSC/SSC
        if 'FSC-A' in pipeline.channels and 'SSC-A' in pipeline.channels:
            pipeline.gate_rectangle(
                'FSC-A', 'SSC-A',
                x_min=30000, x_max=150000,
                y_min=0, y_max=100000,
                parent_gate='singlets'
            )
        
        # CD3+ T cells
        cd3_channels = [ch for ch in pipeline.channels if 'CD3' in ch.upper()]
        if cd3_channels:
            pipeline.gate_gmm_1d(
                cd3_channels[0],
                n_components=2,
                select_component='positive',
                parent_gate='singlets_FSC-A_SSC-A_rect'
            )
        
        # CD4/CD8 quadrants
        cd4_channels = [ch for ch in pipeline.channels if 'CD4' in ch.upper()]
        cd8_channels = [ch for ch in pipeline.channels if 'CD8' in ch.upper()]
        
        if cd4_channels and cd8_channels:
            cd3_gate = [g for g in pipeline.gates.keys() if 'CD3' in g and 'positive' in g]
            if cd3_gate:
                pipeline.gate_quadrants(
                    cd4_channels[0],
                    cd8_channels[0],
                    parent_gate=cd3_gate[0]
                )
    
    def compare_populations(self) -> pd.DataFrame:
        """
        Compare les populations entre tous les échantillons
        
        Returns:
            DataFrame comparatif
        """
        print("\n📊 Comparaison des populations entre échantillons...")
        
        all_stats = []
        
        for sample_name, pipeline in self.pipelines.items():
            stats = pipeline.stats.copy()
            stats['Sample'] = sample_name
            all_stats.append(stats)
        
        if not all_stats:
            print("⚠️  Aucune statistique disponible")
            return pd.DataFrame()
        
        self.comparative_stats = pd.concat(all_stats, ignore_index=True)
        
        # Pivot pour avoir échantillons en colonnes
        pivot_table = self.comparative_stats.pivot_table(
            index='Population',
            columns='Sample',
            values=['Count', 'Percentage_of_total'],
            aggfunc='first'
        )
        
        print(f"✓ Comparaison générée pour {len(self.pipelines)} échantillons")
        
        return pivot_table
    
    def plot_comparative_barplot(self, 
                                 populations: Optional[List[str]] = None,
                                 save_path: Optional[str] = None):
        """
        Graphique en barres comparatif des populations
        
        Args:
            populations: Liste des populations à comparer (None = toutes)
            save_path: Chemin de sauvegarde
        """
        if self.comparative_stats is None:
            print("⚠️  Exécuter compare_populations() d'abord")
            return
        
        # Filtrer les populations si spécifié
        if populations:
            plot_data = self.comparative_stats[
                self.comparative_stats['Population'].isin(populations)
            ]
        else:
            plot_data = self.comparative_stats
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Barplot
        plot_data_pivot = plot_data.pivot(
            index='Sample',
            columns='Population',
            values='Percentage_of_total'
        )
        
        plot_data_pivot.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_xlabel('Échantillon', fontsize=12)
        ax.set_ylabel('Pourcentage du total (%)', fontsize=12)
        ax.set_title('Comparaison des populations entre échantillons', fontsize=14, fontweight='bold')
        ax.legend(title='Population', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_comparative_excel(self, output_path: str):
        """
        Exporte l'analyse comparative dans Excel
        
        Args:
            output_path: Chemin du fichier Excel
        """
        print(f"\n💾 Export comparatif vers: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Feuille 1: Vue d'ensemble
            overview = pd.DataFrame([
                {
                    'Sample': name,
                    'Total_events': len(pipeline.data),
                    'Number_of_populations': len(pipeline.gates),
                    'Channels': len(pipeline.channels)
                }
                for name, pipeline in self.pipelines.items()
            ])
            overview.to_excel(writer, sheet_name='Overview', index=False)
            
            # Feuille 2: Toutes les statistiques
            if self.comparative_stats is not None:
                self.comparative_stats.to_excel(writer, sheet_name='All_Statistics', index=False)
            
            # Feuille 3: Tableau pivot des comptages
            if self.comparative_stats is not None:
                pivot_counts = self.comparative_stats.pivot_table(
                    index='Population',
                    columns='Sample',
                    values='Count',
                    aggfunc='first'
                )
                pivot_counts.to_excel(writer, sheet_name='Population_Counts')
            
            # Feuille 4: Tableau pivot des pourcentages
            if self.comparative_stats is not None:
                pivot_pct = self.comparative_stats.pivot_table(
                    index='Population',
                    columns='Sample',
                    values='Percentage_of_total',
                    aggfunc='first'
                )
                pivot_pct.to_excel(writer, sheet_name='Population_Percentages')
            
            print("  ✓ Feuilles créées: Overview, All_Statistics, Population_Counts, Population_Percentages")
        
        print(f"✅ Export comparatif terminé")


class AdvancedGatingStrategies:
    """
    Stratégies de gating avancées pour cas complexes
    """
    
    @staticmethod
    def gate_live_dead(pipeline: FCSGatingPipeline,
                       viability_channel: str,
                       threshold: Optional[float] = None,
                       parent_gate: Optional[str] = None) -> pd.Series:
        """
        Gating Live/Dead basé sur marqueur de viabilité
        
        Args:
            pipeline: Pipeline FCS
            viability_channel: Canal du marqueur (ex: 'Live-Dead Aqua')
            threshold: Seuil (None = GMM automatique)
            parent_gate: Gate parent
            
        Returns:
            Boolean mask des cellules vivantes
        """
        if threshold is None:
            # Utiliser GMM pour déterminer automatiquement
            return pipeline.gate_gmm_1d(
                viability_channel,
                n_components=2,
                select_component='negative',  # Vivantes = faible signal
                parent_gate=parent_gate
            )
        else:
            # Seuillage manuel
            parent_mask = pipeline.gates.get(parent_gate, pd.Series(True, index=pipeline.data.index))
            gate = (pipeline.data[viability_channel] < threshold) & parent_mask
            
            gate_name = f"live_cells" if not parent_gate else f"{parent_gate}_live"
            pipeline.gates[gate_name] = gate
            pipeline.populations[gate_name] = pipeline.data[gate].copy()
            
            return gate
    
    @staticmethod
    def gate_cd45_ssc_leukocytes(pipeline: FCSGatingPipeline,
                                 cd45_channel: str,
                                 ssc_channel: str = 'SSC-A',
                                 parent_gate: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Gating CD45/SSC pour identifier les sous-populations leucocytaires
        (Lymphocytes, Monocytes, Granulocytes)
        
        Args:
            pipeline: Pipeline FCS
            cd45_channel: Canal CD45
            ssc_channel: Canal SSC
            parent_gate: Gate parent
            
        Returns:
            Dictionnaire des gates de sous-populations
        """
        print(f"🔍 Gating leucocytes (CD45/{ssc_channel})...")
        
        parent_mask = pipeline.gates.get(parent_gate, pd.Series(True, index=pipeline.data.index))
        
        # Zones typiques (à ajuster selon le panel)
        lympho_gate = (
            (pipeline.data[cd45_channel] > 3.5) &
            (pipeline.data[ssc_channel] < 2.5) &
            parent_mask
        )
        
        mono_gate = (
            (pipeline.data[cd45_channel] > 3.0) &
            (pipeline.data[ssc_channel] > 2.5) &
            (pipeline.data[ssc_channel] < 3.5) &
            parent_mask
        )
        
        gran_gate = (
            (pipeline.data[cd45_channel] > 2.5) &
            (pipeline.data[ssc_channel] > 3.5) &
            parent_mask
        )
        
        gates = {
            'lymphocytes': lympho_gate,
            'monocytes': mono_gate,
            'granulocytes': gran_gate
        }
        
        for name, mask in gates.items():
            pipeline.gates[name] = mask
            pipeline.populations[name] = pipeline.data[mask].copy()
            pct = (mask.sum() / parent_mask.sum()) * 100
            print(f"  ✓ {name}: {mask.sum():,} ({pct:.1f}%)")
        
        return gates
    
    @staticmethod
    def gate_memory_phenotype(pipeline: FCSGatingPipeline,
                              cd45ra_channel: str,
                              ccr7_channel: str,
                              parent_gate: str) -> Dict[str, pd.Series]:
        """
        Gating phénotype mémoire T (Naive, CM, EM, TEMRA)
        
        Args:
            pipeline: Pipeline FCS
            cd45ra_channel: Canal CD45RA
            ccr7_channel: Canal CCR7
            parent_gate: Gate des cellules T (ex: CD3+)
            
        Returns:
            Dictionnaire des phénotypes mémoire
        """
        print(f"🔍 Phénotypage mémoire T (CD45RA/CCR7)...")
        
        parent_mask = pipeline.gates[parent_gate]
        parent_data = pipeline.data[parent_mask]
        
        # Détermination des seuils (médiane ou GMM)
        ra_thresh = parent_data[cd45ra_channel].median()
        ccr7_thresh = parent_data[ccr7_channel].median()
        
        # Définition des sous-populations
        naive = (
            (pipeline.data[cd45ra_channel] > ra_thresh) &
            (pipeline.data[ccr7_channel] > ccr7_thresh) &
            parent_mask
        )  # CD45RA+ CCR7+
        
        cm = (
            (pipeline.data[cd45ra_channel] <= ra_thresh) &
            (pipeline.data[ccr7_channel] > ccr7_thresh) &
            parent_mask
        )  # CD45RA- CCR7+ (Central Memory)
        
        em = (
            (pipeline.data[cd45ra_channel] <= ra_thresh) &
            (pipeline.data[ccr7_channel] <= ccr7_thresh) &
            parent_mask
        )  # CD45RA- CCR7- (Effector Memory)
        
        temra = (
            (pipeline.data[cd45ra_channel] > ra_thresh) &
            (pipeline.data[ccr7_channel] <= ccr7_thresh) &
            parent_mask
        )  # CD45RA+ CCR7- (TEMRA)
        
        phenotypes = {
            f'{parent_gate}_Naive': naive,
            f'{parent_gate}_CM': cm,
            f'{parent_gate}_EM': em,
            f'{parent_gate}_TEMRA': temra
        }
        
        for name, mask in phenotypes.items():
            pipeline.gates[name] = mask
            pipeline.populations[name] = pipeline.data[mask].copy()
            pct = (mask.sum() / parent_mask.sum()) * 100
            print(f"  ✓ {name}: {mask.sum():,} ({pct:.1f}%)")
        
        return phenotypes


def example_batch_analysis():
    """
    Exemple d'utilisation pour analyse par lot
    """
    # Liste de fichiers FCS
    fcs_files = [
        'sample1.fcs',
        'sample2.fcs',
        'sample3.fcs'
    ]
    
    # Noms personnalisés
    sample_names = ['Control', 'Treatment_A', 'Treatment_B']
    
    # Initialisation
    batch = BatchFCSAnalysis(fcs_files, sample_names)
    
    # Exécution du pipeline
    pipelines = batch.run_standard_pipeline(
        compensate=True,
        transform='logicle',
        gate_strategy='standard'
    )
    
    # Comparaison
    comparison = batch.compare_populations()
    print("\n📊 Résultats comparatifs:")
    print(comparison)
    
    # Visualisation
    batch.plot_comparative_barplot(
        save_path='./results/comparative_populations.png'
    )
    
    # Export
    batch.export_comparative_excel('./results/batch_analysis.xlsx')
    
    return batch


def example_advanced_tcell_panel():
    """
    Exemple d'analyse avancée: Panel T cells complet
    """
    # Charger l'échantillon
    pipeline = FCSGatingPipeline(
        'tcell_panel.fcs',
        compensate=True,
        transform='logicle'
    )
    
    # 1. Singlets
    pipeline.gate_singlets_fsc_ssc()
    
    # 2. Live/Dead
    AdvancedGatingStrategies.gate_live_dead(
        pipeline,
        viability_channel='Live-Dead',
        parent_gate='singlets'
    )
    
    # 3. CD45+ leucocytes
    AdvancedGatingStrategies.gate_cd45_ssc_leukocytes(
        pipeline,
        cd45_channel='CD45',
        parent_gate='singlets_live_cells'
    )
    
    # 4. CD3+ T cells
    pipeline.gate_gmm_1d(
        'CD3',
        n_components=2,
        select_component='positive',
        parent_gate='lymphocytes'
    )
    
    # 5. CD4/CD8 subsets
    quadrants = pipeline.gate_quadrants(
        'CD4',
        'CD8',
        parent_gate='lymphocytes_CD3_positive'
    )
    
    # 6. Memory phenotype sur CD4+ et CD8+
    for subset in ['lymphocytes_CD3_positive_CD4+CD8-', 'lymphocytes_CD3_positive_CD4-CD8+']:
        if subset in pipeline.gates:
            AdvancedGatingStrategies.gate_memory_phenotype(
                pipeline,
                cd45ra_channel='CD45RA',
                ccr7_channel='CCR7',
                parent_gate=subset
            )
    
    # Statistiques et export
    stats = pipeline.compute_statistics()
    pipeline.export_to_excel('./results/tcell_panel_analysis.xlsx')
    
    return pipeline, stats


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODULE DE WORKFLOWS AVANCÉS FACS")
    print("="*60)
    print("\nFonctionnalités disponibles:")
    print("  • Analyse par lot (BatchFCSAnalysis)")
    print("  • Comparaison multi-échantillons")
    print("  • Stratégies de gating avancées")
    print("  • Phénotypage immunitaire complet")
    print("\nConsultez les fonctions example_* pour des exemples d'utilisation")
    print("="*60 + "\n")
