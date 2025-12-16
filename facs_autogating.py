#!/usr/bin/env python3
"""
Module d'automatisation du gating pour cytométrie en flux
Inspiré de FlowKit, OpenCyto et FlowWorkspace
Auteur: Pipeline généralisable pour analyses FACS
"""

import flowkit as fk
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class FCSGatingPipeline:
    """
    Pipeline automatisé de gating pour fichiers FCS
    
    Méthodes de gating supportées:
    - Gating basé sur les quantiles (debris removal)
    - Gaussian Mixture Models (GMM) pour populations multimodales
    - DBSCAN pour gating basé sur la densité
    - Gating manuel avec coordonnées définies
    """
    
    def __init__(self, fcs_path: str, compensate: bool = True, transform: str = 'logicle'):
        """
        Initialisation du pipeline
        
        Args:
            fcs_path: Chemin vers le fichier FCS
            compensate: Appliquer la compensation spectrale si disponible
            transform: Type de transformation ('logicle', 'asinh', 'hyperlog', None)
        """
        self.fcs_path = fcs_path
        self.sample = fk.Sample(fcs_path)
        self.compensate = compensate
        self.transform_type = transform
        self.gates = {}
        self.populations = {}
        self.stats = {}
        
        # Appliquer compensation si disponible
        if compensate and self.sample.compensation_matrix is not None:
            self.sample.apply_compensation(self.sample.compensation_matrix)
        
        # Appliquer transformation
        if transform:
            self._apply_transform(transform)
        
        # Extraire les données
        self.data = pd.DataFrame(
            self.sample.as_dataframe(source='xform' if transform else 'raw')
        )
        self.channels = list(self.data.columns)
        
    def _apply_transform(self, transform_type: str):
        """Applique une transformation aux données"""
        xform = None
        if transform_type == 'logicle':
            xform = fk.transforms.LogicleTransform('logicle', param_t=262144, param_w=0.5, param_m=4.5, param_a=0)
        elif transform_type == 'asinh':
            xform = fk.transforms.AsinhTransform('asinh', param_t=262144, param_m=4.5, param_a=0)
        elif transform_type == 'hyperlog':
            xform = fk.transforms.HyperlogTransform('hyperlog', param_t=262144, param_w=0.5, param_m=4.5, param_a=0)
        
        if xform:
            self.sample.apply_transform(xform)
    
    def gate_singlets_fsc_ssc(self, 
                               fsc_channel: str = 'FSC-A',
                               fsc_h_channel: str = 'FSC-H',
                               method: str = 'linear_fit',
                               threshold: float = 2.5) -> pd.Series:
        """
        Gate pour sélectionner les singlets (cellules uniques)
        
        Args:
            fsc_channel: Canal FSC-A (area)
            fsc_h_channel: Canal FSC-H (height)
            method: 'linear_fit' ou 'ratio'
            threshold: Seuil en écarts-types pour la sélection
            
        Returns:
            Boolean mask des singlets
        """
        print(f"🔍 Gating singlets ({fsc_channel} vs {fsc_h_channel})...")
        
        if method == 'linear_fit':
            # Régression linéaire pour identifier la population principale
            from sklearn.linear_model import RANSACRegressor
            
            X = self.data[fsc_channel].values.reshape(-1, 1)
            y = self.data[fsc_h_channel].values
            
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X, y)
            
            # Calcul des résidus
            y_pred = ransac.predict(X)
            residuals = np.abs(y - y_pred)
            threshold_value = np.median(residuals) + threshold * stats.median_abs_deviation(residuals)
            
            gate = residuals < threshold_value
            
        else:  # ratio method
            ratio = self.data[fsc_h_channel] / (self.data[fsc_channel] + 1e-10)
            q1, q3 = np.percentile(ratio, [25, 75])
            iqr = q3 - q1
            gate = (ratio > q1 - threshold * iqr) & (ratio < q3 + threshold * iqr)
        
        self.gates['singlets'] = gate
        self.populations['singlets'] = self.data[gate].copy()
        
        pct = (gate.sum() / len(gate)) * 100
        print(f"✓ Singlets: {gate.sum():,} / {len(gate):,} événements ({pct:.1f}%)")
        
        return gate
    
    def gate_debris_removal(self,
                           fsc_channel: str = 'FSC-A',
                           ssc_channel: str = 'SSC-A',
                           percentile_low: float = 2,
                           parent_gate: Optional[str] = None) -> pd.Series:
        """
        Suppression des débris cellulaires par seuillage des FSC/SSC
        
        Args:
            fsc_channel: Canal Forward Scatter
            ssc_channel: Canal Side Scatter
            percentile_low: Percentile inférieur pour le seuil
            parent_gate: Gate parent à appliquer (None = toutes les cellules)
            
        Returns:
            Boolean mask des cellules viables
        """
        print(f"🔍 Suppression des débris ({fsc_channel}, {ssc_channel})...")
        
        # Données parentes
        if parent_gate and parent_gate in self.gates:
            parent_data = self.data[self.gates[parent_gate]]
            parent_mask = self.gates[parent_gate]
        else:
            parent_data = self.data
            parent_mask = pd.Series(True, index=self.data.index)
        
        # Seuils basés sur les percentiles
        fsc_threshold = np.percentile(parent_data[fsc_channel], percentile_low)
        ssc_threshold = np.percentile(parent_data[ssc_channel], percentile_low)
        
        gate = (self.data[fsc_channel] > fsc_threshold) & \
               (self.data[ssc_channel] > ssc_threshold) & \
               parent_mask
        
        gate_name = 'cells' if parent_gate is None else f'{parent_gate}_viable'
        self.gates[gate_name] = gate
        self.populations[gate_name] = self.data[gate].copy()
        
        pct = (gate.sum() / parent_mask.sum()) * 100
        print(f"✓ Cellules viables: {gate.sum():,} / {parent_mask.sum():,} événements ({pct:.1f}%)")
        
        return gate
    
    def gate_gmm_1d(self,
                    channel: str,
                    n_components: int = 2,
                    select_component: str = 'positive',
                    parent_gate: Optional[str] = None) -> pd.Series:
        """
        Gating 1D par Gaussian Mixture Model
        Idéal pour populations bimodales (CD4+/-, CD8+/-, etc.)
        
        Args:
            channel: Canal à analyser
            n_components: Nombre de composantes gaussiennes
            select_component: 'positive' (composante haute), 'negative' (composante basse), 'all'
            parent_gate: Gate parent
            
        Returns:
            Boolean mask de la population sélectionnée
        """
        print(f"🔍 GMM 1D sur {channel} ({n_components} composantes)...")
        
        # Données parentes
        if parent_gate and parent_gate in self.gates:
            parent_data = self.data[self.gates[parent_gate]]
            parent_mask = self.gates[parent_gate]
        else:
            parent_data = self.data
            parent_mask = pd.Series(True, index=self.data.index)
        
        # Fit GMM
        X = parent_data[channel].values.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        
        # Prédiction des labels
        labels = gmm.predict(X)
        
        # Sélection de la composante
        means = gmm.means_.flatten()
        if select_component == 'positive':
            selected_label = np.argmax(means)  # Composante avec la plus haute moyenne
        elif select_component == 'negative':
            selected_label = np.argmin(means)  # Composante avec la plus basse moyenne
        else:
            selected_label = None
        
        # Création du gate
        gate = parent_mask.copy()
        gate[parent_mask] = (labels == selected_label) if selected_label is not None else True
        
        gate_name = f"{channel}_{select_component}" if parent_gate is None else f"{parent_gate}_{channel}_{select_component}"
        self.gates[gate_name] = gate
        self.populations[gate_name] = self.data[gate].copy()
        
        pct = (gate.sum() / parent_mask.sum()) * 100
        print(f"✓ {gate_name}: {gate.sum():,} / {parent_mask.sum():,} événements ({pct:.1f}%)")
        
        return gate
    
    def gate_gmm_2d(self,
                    channel_x: str,
                    channel_y: str,
                    n_components: int = 3,
                    select_components: Optional[List[int]] = None,
                    parent_gate: Optional[str] = None) -> pd.Series:
        """
        Gating 2D par Gaussian Mixture Model
        Idéal pour analyses multiparamétriques (CD4 vs CD8, etc.)
        
        Args:
            channel_x: Premier canal
            channel_y: Deuxième canal
            n_components: Nombre de composantes
            select_components: Liste des indices de composantes à sélectionner (None = toutes)
            parent_gate: Gate parent
            
        Returns:
            Boolean mask de la population
        """
        print(f"🔍 GMM 2D sur {channel_x} vs {channel_y} ({n_components} composantes)...")
        
        # Données parentes
        if parent_gate and parent_gate in self.gates:
            parent_data = self.data[self.gates[parent_gate]]
            parent_mask = self.gates[parent_gate]
        else:
            parent_data = self.data
            parent_mask = pd.Series(True, index=self.data.index)
        
        # Fit GMM
        X = parent_data[[channel_x, channel_y]].values
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(X)
        
        # Prédiction
        labels = gmm.predict(X)
        
        # Sélection des composantes
        if select_components is None:
            selected_mask = np.ones(len(labels), dtype=bool)
        else:
            selected_mask = np.isin(labels, select_components)
        
        gate = parent_mask.copy()
        gate[parent_mask] = selected_mask
        
        gate_name = f"{channel_x}_{channel_y}_pop" if parent_gate is None else f"{parent_gate}_{channel_x}_{channel_y}"
        self.gates[gate_name] = gate
        self.populations[gate_name] = self.data[gate].copy()
        
        pct = (gate.sum() / parent_mask.sum()) * 100
        print(f"✓ {gate_name}: {gate.sum():,} / {parent_mask.sum():,} événements ({pct:.1f}%)")
        
        return gate
    
    def gate_rectangle(self,
                       channel_x: str,
                       channel_y: str,
                       x_min: float,
                       x_max: float,
                       y_min: float,
                       y_max: float,
                       parent_gate: Optional[str] = None) -> pd.Series:
        """
        Gating rectangulaire manuel
        
        Args:
            channel_x, channel_y: Canaux
            x_min, x_max, y_min, y_max: Coordonnées du rectangle
            parent_gate: Gate parent
            
        Returns:
            Boolean mask
        """
        print(f"🔍 Gate rectangulaire sur {channel_x} vs {channel_y}...")
        
        if parent_gate and parent_gate in self.gates:
            parent_mask = self.gates[parent_gate]
        else:
            parent_mask = pd.Series(True, index=self.data.index)
        
        gate = (self.data[channel_x] >= x_min) & \
               (self.data[channel_x] <= x_max) & \
               (self.data[channel_y] >= y_min) & \
               (self.data[channel_y] <= y_max) & \
               parent_mask
        
        gate_name = f"{channel_x}_{channel_y}_rect" if parent_gate is None else f"{parent_gate}_{channel_x}_{channel_y}_rect"
        self.gates[gate_name] = gate
        self.populations[gate_name] = self.data[gate].copy()
        
        pct = (gate.sum() / parent_mask.sum()) * 100
        print(f"✓ {gate_name}: {gate.sum():,} / {parent_mask.sum():,} événements ({pct:.1f}%)")
        
        return gate
    
    def gate_quadrants(self,
                       channel_x: str,
                       channel_y: str,
                       x_threshold: Optional[float] = None,
                       y_threshold: Optional[float] = None,
                       parent_gate: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Gating en quadrants (Q1-Q4)
        Retourne 4 populations: ++, +-, -+, --
        
        Args:
            channel_x, channel_y: Canaux
            x_threshold: Seuil X (None = médiane)
            y_threshold: Seuil Y (None = médiane)
            parent_gate: Gate parent
            
        Returns:
            Dictionnaire de gates {nom: mask}
        """
        print(f"🔍 Quadrants sur {channel_x} vs {channel_y}...")
        
        if parent_gate and parent_gate in self.gates:
            parent_data = self.data[self.gates[parent_gate]]
            parent_mask = self.gates[parent_gate]
        else:
            parent_data = self.data
            parent_mask = pd.Series(True, index=self.data.index)
        
        # Détermination des seuils
        x_thresh = x_threshold if x_threshold is not None else parent_data[channel_x].median()
        y_thresh = y_threshold if y_threshold is not None else parent_data[channel_y].median()
        
        # Création des quadrants
        quadrants = {}
        prefix = f"{parent_gate}_" if parent_gate else ""
        
        # Q1: X+Y+ (haut-droite)
        q1 = (self.data[channel_x] > x_thresh) & (self.data[channel_y] > y_thresh) & parent_mask
        quadrants[f"{prefix}{channel_x}+{channel_y}+"] = q1
        
        # Q2: X-Y+ (haut-gauche)
        q2 = (self.data[channel_x] <= x_thresh) & (self.data[channel_y] > y_thresh) & parent_mask
        quadrants[f"{prefix}{channel_x}-{channel_y}+"] = q2
        
        # Q3: X-Y- (bas-gauche)
        q3 = (self.data[channel_x] <= x_thresh) & (self.data[channel_y] <= y_thresh) & parent_mask
        quadrants[f"{prefix}{channel_x}-{channel_y}-"] = q3
        
        # Q4: X+Y- (bas-droite)
        q4 = (self.data[channel_x] > x_thresh) & (self.data[channel_y] <= y_thresh) & parent_mask
        quadrants[f"{prefix}{channel_x}+{channel_y}-"] = q4
        
        # Stockage
        for name, mask in quadrants.items():
            self.gates[name] = mask
            self.populations[name] = self.data[mask].copy()
            pct = (mask.sum() / parent_mask.sum()) * 100
            print(f"  ✓ {name}: {mask.sum():,} ({pct:.1f}%)")
        
        return quadrants
    
    def compute_statistics(self) -> pd.DataFrame:
        """
        Calcule les statistiques pour toutes les populations
        
        Returns:
            DataFrame avec les statistiques
        """
        print("\n📊 Calcul des statistiques...")
        
        stats_list = []
        
        for gate_name, mask in self.gates.items():
            pop_data = self.data[mask]
            
            stat_row = {
                'Population': gate_name,
                'Count': mask.sum(),
                'Percentage_of_total': (mask.sum() / len(self.data)) * 100,
            }
            
            # Statistiques par canal
            for channel in self.channels:
                if channel in pop_data.columns:
                    stat_row[f'{channel}_mean'] = pop_data[channel].mean()
                    stat_row[f'{channel}_median'] = pop_data[channel].median()
                    stat_row[f'{channel}_std'] = pop_data[channel].std()
            
            stats_list.append(stat_row)
        
        self.stats = pd.DataFrame(stats_list)
        print(f"✓ Statistiques calculées pour {len(self.gates)} populations")
        
        return self.stats
    
    def plot_gates(self,
                   channel_x: str,
                   channel_y: str,
                   gates_to_plot: Optional[List[str]] = None,
                   figsize: Tuple[int, int] = (12, 10),
                   save_path: Optional[str] = None):
        """
        Visualisation des gates sur un scatter plot 2D
        
        Args:
            channel_x, channel_y: Canaux à afficher
            gates_to_plot: Liste des noms de gates à afficher (None = tous)
            figsize: Taille de la figure
            save_path: Chemin pour sauvegarder (None = affichage seulement)
        """
        gates_list = gates_to_plot if gates_to_plot else list(self.gates.keys())
        n_gates = len(gates_list)
        
        if n_gates == 0:
            print("Aucun gate à afficher")
            return
        
        # Organisation en sous-graphiques
        n_cols = min(3, n_gates)
        n_rows = (n_gates + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_gates == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, gate_name in enumerate(gates_list):
            ax = axes[idx]
            mask = self.gates[gate_name]
            
            # Background: toutes les cellules
            ax.scatter(self.data[channel_x], self.data[channel_y],
                      s=1, c='lightgray', alpha=0.3, rasterized=True)
            
            # Foreground: population gatée
            ax.scatter(self.data.loc[mask, channel_x],
                      self.data.loc[mask, channel_y],
                      s=1, c='red', alpha=0.5, rasterized=True)
            
            ax.set_xlabel(channel_x)
            ax.set_ylabel(channel_y)
            pct = (mask.sum() / len(mask)) * 100
            ax.set_title(f"{gate_name}\n{mask.sum():,} events ({pct:.1f}%)")
            ax.grid(True, alpha=0.3)
        
        # Masquer les axes non utilisés
        for idx in range(n_gates, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figure sauvegardée: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_to_excel(self, output_path: str, include_populations: bool = True):
        """
        Exporte les résultats dans un fichier Excel
        
        Args:
            output_path: Chemin du fichier Excel de sortie
            include_populations: Inclure les données brutes de chaque population
        """
        print(f"\n💾 Export vers Excel: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Feuille 1: Résumé des statistiques
            if len(self.stats) > 0:
                self.stats.to_excel(writer, sheet_name='Statistics', index=False)
                print("  ✓ Feuille 'Statistics' créée")
            
            # Feuille 2: Informations du fichier
            metadata = {
                'Fichier': [Path(self.fcs_path).name],
                'Nombre_evenements_total': [len(self.data)],
                'Nombre_canaux': [len(self.channels)],
                'Canaux': [', '.join(self.channels[:10]) + '...' if len(self.channels) > 10 else ', '.join(self.channels)],
                'Compensation': ['Oui' if self.compensate else 'Non'],
                'Transformation': [self.transform_type if self.transform_type else 'Aucune'],
                'Nombre_populations': [len(self.gates)]
            }
            pd.DataFrame(metadata).to_excel(writer, sheet_name='File_Info', index=False)
            print("  ✓ Feuille 'File_Info' créée")
            
            # Feuille 3: Comptage des populations
            pop_counts = pd.DataFrame([
                {
                    'Population': name,
                    'Count': mask.sum(),
                    'Percentage': (mask.sum() / len(self.data)) * 100
                }
                for name, mask in self.gates.items()
            ])
            pop_counts.to_excel(writer, sheet_name='Population_Counts', index=False)
            print("  ✓ Feuille 'Population_Counts' créée")
            
            # Feuilles supplémentaires: Données brutes de chaque population (optionnel)
            if include_populations and len(self.populations) > 0:
                for name, pop_df in list(self.populations.items())[:20]:  # Limiter à 20 pour éviter fichiers trop lourds
                    sheet_name = name[:31]  # Excel limite à 31 caractères
                    pop_df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  ✓ {min(20, len(self.populations))} feuilles de populations créées")
        
        print(f"✅ Export terminé: {output_path}")


def example_standard_workflow(fcs_path: str, output_dir: str = './results'):
    """
    Exemple de workflow standard pour l'analyse FACS
    
    Args:
        fcs_path: Chemin vers le fichier FCS
        output_dir: Répertoire de sortie
    """
    # Création du répertoire de sortie
    Path(output_dir).mkdir(exist_ok=True)
    
    # Nom de base pour les fichiers de sortie
    base_name = Path(fcs_path).stem
    
    print(f"\n{'='*60}")
    print(f"ANALYSE FACS AUTOMATISÉE: {base_name}")
    print(f"{'='*60}\n")
    
    # 1. Chargement et prétraitement
    pipeline = FCSGatingPipeline(
        fcs_path,
        compensate=True,
        transform='logicle'
    )
    
    print(f"\n📁 Fichier chargé: {fcs_path}")
    print(f"📊 {len(pipeline.data):,} événements, {len(pipeline.channels)} canaux")
    print(f"📋 Canaux: {', '.join(pipeline.channels[:8])}{'...' if len(pipeline.channels) > 8 else ''}")
    
    # 2. Stratégie de gating standard
    print(f"\n{'='*60}")
    print("SÉQUENCE DE GATING")
    print(f"{'='*60}\n")
    
    # Étape 1: Singlets
    try:
        pipeline.gate_singlets_fsc_ssc(
            fsc_channel='FSC-A',
            fsc_h_channel='FSC-H',
            method='linear_fit'
        )
    except Exception as e:
        print(f"⚠️  Gating singlets échoué: {e}")
    
    # Étape 2: Debris removal
    try:
        pipeline.gate_debris_removal(
            fsc_channel='FSC-A',
            ssc_channel='SSC-A',
            percentile_low=2,
            parent_gate='singlets'
        )
    except Exception as e:
        print(f"⚠️  Debris removal échoué: {e}")
    
    # Étape 3: Recherche automatique de marqueurs positifs
    # (Exemple: CD3, CD4, CD8, CD19, CD56, etc.)
    marker_channels = [ch for ch in pipeline.channels 
                      if any(marker in ch.upper() for marker in ['CD', 'FITC', 'PE', 'APC', 'BV'])]
    
    print(f"\n🔬 Marqueurs détectés: {len(marker_channels)}")
    for marker in marker_channels[:10]:
        try:
            pipeline.gate_gmm_1d(
                channel=marker,
                n_components=2,
                select_component='positive',
                parent_gate='singlets_viable'
            )
        except Exception as e:
            print(f"  ⚠️  GMM 1D {marker}: {e}")
    
    # 3. Calcul des statistiques
    stats = pipeline.compute_statistics()
    
    # 4. Visualisations
    print(f"\n{'='*60}")
    print("GÉNÉRATION DES VISUALISATIONS")
    print(f"{'='*60}\n")
    
    # Plot FSC/SSC
    if 'FSC-A' in pipeline.channels and 'SSC-A' in pipeline.channels:
        pipeline.plot_gates(
            'FSC-A', 'SSC-A',
            gates_to_plot=['singlets', 'singlets_viable'] if 'singlets_viable' in pipeline.gates else ['singlets'],
            save_path=f"{output_dir}/{base_name}_FSC_SSC.png"
        )
    
    # 5. Export Excel
    excel_path = f"{output_dir}/{base_name}_analysis.xlsx"
    pipeline.export_to_excel(excel_path, include_populations=True)
    
    print(f"\n{'='*60}")
    print(f"✅ ANALYSE TERMINÉE")
    print(f"{'='*60}")
    print(f"📊 {len(pipeline.gates)} populations identifiées")
    print(f"📁 Résultats dans: {output_dir}")
    print(f"{'='*60}\n")
    
    return pipeline, stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python facs_autogating.py <fichier.fcs> [output_dir]")
        sys.exit(1)
    
    fcs_file = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else './results'
    
    pipeline, stats = example_standard_workflow(fcs_file, output_directory)
    print("\n📈 Aperçu des statistiques:")
    print(stats[['Population', 'Count', 'Percentage_of_total']].to_string(index=False))
