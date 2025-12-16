#!/usr/bin/env python3
"""
Fonctions utilitaires pour le pipeline FACS
Helpers, validation, QC, et formatage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import flowkit as fk
import warnings


class FCSValidator:
    """
    Validation et contrôle qualité des fichiers FCS
    """
    
    @staticmethod
    def validate_fcs_file(fcs_path: str) -> Dict[str, any]:
        """
        Valide un fichier FCS et retourne un rapport de validation
        
        Args:
            fcs_path: Chemin vers le fichier FCS
            
        Returns:
            Dictionnaire avec résultats de validation
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Vérifier existence
        if not Path(fcs_path).exists():
            validation['is_valid'] = False
            validation['errors'].append(f"Fichier non trouvé: {fcs_path}")
            return validation
        
        # Essayer de charger
        try:
            sample = fk.Sample(fcs_path)
            validation['info']['event_count'] = sample.event_count
            validation['info']['channel_count'] = len(sample.channels)
            validation['info']['channels'] = sample.pnn_labels
            
            # Vérifier nombre d'événements
            if sample.event_count < 1000:
                validation['warnings'].append(
                    f"Nombre d'événements faible: {sample.event_count}"
                )
            
            # Vérifier canaux standards
            standard_channels = ['FSC-A', 'SSC-A', 'FSC-H']
            missing = [ch for ch in standard_channels 
                      if not any(ch in pnn for pnn in sample.pnn_labels)]
            
            if missing:
                validation['warnings'].append(
                    f"Canaux standards manquants: {', '.join(missing)}"
                )
            
            # Vérifier compensation
            if sample.compensation_matrix is None:
                validation['warnings'].append(
                    "Pas de matrice de compensation trouvée"
                )
            else:
                validation['info']['has_compensation'] = True
            
            validation['info']['file_size_mb'] = Path(fcs_path).stat().st_size / (1024**2)
            
        except Exception as e:
            validation['is_valid'] = False
            validation['errors'].append(f"Erreur lors du chargement: {str(e)}")
        
        return validation
    
    @staticmethod
    def batch_validate_fcs(fcs_files: List[str]) -> pd.DataFrame:
        """
        Validation par lot de fichiers FCS
        
        Args:
            fcs_files: Liste des chemins de fichiers
            
        Returns:
            DataFrame avec résultats de validation
        """
        results = []
        
        for fcs_file in fcs_files:
            val = FCSValidator.validate_fcs_file(fcs_file)
            
            result = {
                'File': Path(fcs_file).name,
                'Valid': val['is_valid'],
                'Event_Count': val['info'].get('event_count', 0),
                'Channel_Count': val['info'].get('channel_count', 0),
                'Has_Compensation': val['info'].get('has_compensation', False),
                'File_Size_MB': val['info'].get('file_size_mb', 0),
                'Errors': '; '.join(val['errors']) if val['errors'] else 'None',
                'Warnings': '; '.join(val['warnings']) if val['warnings'] else 'None'
            }
            results.append(result)
        
        return pd.DataFrame(results)


class ChannelDetector:
    """
    Détection automatique de canaux et marqueurs
    """
    
    # Dictionnaire de marqueurs courants
    COMMON_MARKERS = {
        'T_CELLS': ['CD3', 'CD4', 'CD8', 'CD45RA', 'CCR7', 'CD25', 'CD127'],
        'B_CELLS': ['CD19', 'CD20', 'IgD', 'CD27', 'CD38'],
        'NK_CELLS': ['CD56', 'CD16', 'NKG2D', 'NKp46'],
        'MONOCYTES': ['CD14', 'CD16', 'HLA-DR'],
        'ACTIVATION': ['CD69', 'CD25', 'HLA-DR', 'CD38'],
        'MEMORY': ['CD45RA', 'CD45RO', 'CCR7', 'CD62L', 'CD27'],
        'CYTOKINES': ['IFNg', 'TNF', 'IL2', 'IL4', 'IL17'],
        'VIABILITY': ['LIVE', 'DEAD', 'PI', '7AAD', 'DAPI'],
        'PROLIFERATION': ['Ki67', 'BrdU', 'EdU']
    }
    
    @staticmethod
    def detect_channels(pipeline) -> Dict[str, List[str]]:
        """
        Détecte automatiquement les types de canaux présents
        
        Args:
            pipeline: Instance de FCSGatingPipeline
            
        Returns:
            Dictionnaire {catégorie: [canaux]}
        """
        detected = {}
        channels = [ch.upper() for ch in pipeline.channels]
        
        for category, markers in ChannelDetector.COMMON_MARKERS.items():
            found = []
            for marker in markers:
                matching = [ch for ch in pipeline.channels 
                           if marker.upper() in ch.upper()]
                found.extend(matching)
            
            if found:
                detected[category] = found
        
        # Détecter FSC/SSC
        fsc_channels = [ch for ch in pipeline.channels if 'FSC' in ch.upper()]
        ssc_channels = [ch for ch in pipeline.channels if 'SSC' in ch.upper()]
        
        if fsc_channels:
            detected['FSC'] = fsc_channels
        if ssc_channels:
            detected['SSC'] = ssc_channels
        
        return detected
    
    @staticmethod
    def suggest_gating_strategy(detected_channels: Dict[str, List[str]]) -> List[str]:
        """
        Suggère une stratégie de gating basée sur les canaux détectés
        
        Args:
            detected_channels: Résultat de detect_channels()
            
        Returns:
            Liste de suggestions de gating
        """
        suggestions = []
        
        # QC standard
        if 'FSC' in detected_channels:
            suggestions.append("1. Gate singlets avec FSC-A vs FSC-H")
        
        if 'VIABILITY' in detected_channels:
            suggestions.append(
                f"2. Gate cellules vivantes avec {detected_channels['VIABILITY'][0]}"
            )
        
        if 'FSC' in detected_channels and 'SSC' in detected_channels:
            suggestions.append("3. Gate débris avec FSC-A vs SSC-A")
        
        # Marqueurs spécifiques
        if 'T_CELLS' in detected_channels:
            cd3 = [ch for ch in detected_channels['T_CELLS'] if 'CD3' in ch.upper()]
            if cd3:
                suggestions.append(f"4. Gate T cells (CD3+) avec {cd3[0]}")
            
            cd4 = [ch for ch in detected_channels['T_CELLS'] if 'CD4' in ch.upper()]
            cd8 = [ch for ch in detected_channels['T_CELLS'] if 'CD8' in ch.upper()]
            
            if cd4 and cd8:
                suggestions.append(
                    f"5. Quadrants CD4/CD8 avec {cd4[0]} vs {cd8[0]}"
                )
        
        if 'MEMORY' in detected_channels:
            cd45ra = [ch for ch in detected_channels['MEMORY'] if 'CD45RA' in ch.upper()]
            ccr7 = [ch for ch in detected_channels['MEMORY'] if 'CCR7' in ch.upper()]
            
            if cd45ra and ccr7:
                suggestions.append(
                    f"6. Phénotype mémoire avec {cd45ra[0]} vs {ccr7[0]}"
                )
        
        if 'B_CELLS' in detected_channels:
            cd19 = [ch for ch in detected_channels['B_CELLS'] if 'CD19' in ch.upper()]
            if cd19:
                suggestions.append(f"Gate B cells (CD19+) avec {cd19[0]}")
        
        if 'NK_CELLS' in detected_channels:
            cd56 = [ch for ch in detected_channels['NK_CELLS'] if 'CD56' in ch.upper()]
            cd16 = [ch for ch in detected_channels['NK_CELLS'] if 'CD16' in ch.upper()]
            if cd56 and cd16:
                suggestions.append(
                    f"Gate NK cells avec {cd56[0]} vs {cd16[0]}"
                )
        
        return suggestions


class StatisticsExporter:
    """
    Export et formatage avancé des statistiques
    """
    
    @staticmethod
    def create_summary_table(pipeline) -> pd.DataFrame:
        """
        Crée un tableau récapitulatif formaté
        
        Args:
            pipeline: Instance de FCSGatingPipeline
            
        Returns:
            DataFrame formaté
        """
        summary = []
        
        total_events = len(pipeline.data)
        
        for gate_name, mask in pipeline.gates.items():
            count = mask.sum()
            pct_total = (count / total_events) * 100
            
            # Déterminer le parent
            parent = 'Total'
            parent_count = total_events
            
            # Chercher le parent potentiel
            for potential_parent in pipeline.gates.keys():
                if potential_parent in gate_name and potential_parent != gate_name:
                    if len(potential_parent) < len(gate_name):
                        parent = potential_parent
                        parent_count = pipeline.gates[potential_parent].sum()
                        break
            
            pct_parent = (count / parent_count) * 100 if parent_count > 0 else 0
            
            summary.append({
                'Population': gate_name,
                'Count': f"{count:,}",
                '%_of_Total': f"{pct_total:.2f}",
                'Parent': parent,
                '%_of_Parent': f"{pct_parent:.2f}"
            })
        
        return pd.DataFrame(summary)
    
    @staticmethod
    def calculate_mfi(pipeline, channels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calcule la Mean Fluorescence Intensity (MFI) pour chaque population
        
        Args:
            pipeline: Instance de FCSGatingPipeline
            channels: Liste de canaux (None = tous)
            
        Returns:
            DataFrame avec MFI par population et canal
        """
        if channels is None:
            # Exclure FSC/SSC/Time
            channels = [ch for ch in pipeline.channels 
                       if not any(x in ch.upper() for x in ['FSC', 'SSC', 'TIME'])]
        
        mfi_data = []
        
        for gate_name, mask in pipeline.gates.items():
            pop_data = pipeline.data[mask]
            
            row = {'Population': gate_name}
            
            for channel in channels:
                if channel in pop_data.columns:
                    mfi = pop_data[channel].mean()
                    row[f'{channel}_MFI'] = mfi
            
            mfi_data.append(row)
        
        return pd.DataFrame(mfi_data)


class QualityControl:
    """
    Contrôles qualité pour données de cytométrie
    """
    
    @staticmethod
    def check_time_drift(pipeline, time_channel: str = 'Time') -> Dict:
        """
        Vérifie le drift temporel dans l'acquisition
        
        Args:
            pipeline: Instance de FCSGatingPipeline
            time_channel: Nom du canal temporel
            
        Returns:
            Résultats du contrôle de drift
        """
        if time_channel not in pipeline.data.columns:
            return {'status': 'error', 'message': f'Canal {time_channel} non trouvé'}
        
        # Diviser en bins temporels
        n_bins = 10
        time_data = pipeline.data[time_channel]
        time_bins = pd.qcut(time_data, q=n_bins, labels=False, duplicates='drop')
        
        # Calculer statistiques par bin
        drift_stats = []
        
        for marker in pipeline.channels:
            if marker == time_channel or 'FSC' in marker or 'SSC' in marker:
                continue
            
            means_by_bin = [
                pipeline.data[time_bins == i][marker].mean()
                for i in range(n_bins)
            ]
            
            # Coefficient de variation
            cv = np.std(means_by_bin) / np.mean(means_by_bin) * 100
            
            drift_stats.append({
                'Channel': marker,
                'CV_across_time': cv,
                'Has_drift': cv > 10  # Seuil arbitraire de 10%
            })
        
        drift_df = pd.DataFrame(drift_stats)
        
        return {
            'status': 'success',
            'n_drifting_channels': drift_df['Has_drift'].sum(),
            'details': drift_df
        }
    
    @staticmethod
    def check_doublet_rate(pipeline, singlets_gate: str = 'singlets') -> Dict:
        """
        Calcule le taux de doublets
        
        Args:
            pipeline: Instance de FCSGatingPipeline
            singlets_gate: Nom du gate des singlets
            
        Returns:
            Statistiques sur les doublets
        """
        if singlets_gate not in pipeline.gates:
            return {'status': 'error', 'message': f'Gate {singlets_gate} non trouvé'}
        
        singlets = pipeline.gates[singlets_gate].sum()
        total = len(pipeline.data)
        doublets = total - singlets
        doublet_rate = (doublets / total) * 100
        
        # Évaluation
        if doublet_rate < 5:
            quality = 'Excellent'
        elif doublet_rate < 10:
            quality = 'Bon'
        elif doublet_rate < 20:
            quality = 'Acceptable'
        else:
            quality = 'Mauvais - Optimisation recommandée'
        
        return {
            'status': 'success',
            'doublet_rate': doublet_rate,
            'singlets': singlets,
            'doublets': doublets,
            'quality_assessment': quality
        }


def auto_suggest_workflow(fcs_path: str) -> str:
    """
    Analyse un fichier FCS et suggère un workflow adapté
    
    Args:
        fcs_path: Chemin du fichier FCS
        
    Returns:
        Code Python suggéré
    """
    # Charger temporairement
    from facs_autogating import FCSGatingPipeline
    
    temp_pipeline = FCSGatingPipeline(fcs_path, compensate=False, transform=None)
    
    # Détecter les canaux
    detected = ChannelDetector.detect_channels(temp_pipeline)
    
    # Générer le code
    code = f"""
# Workflow suggéré pour: {Path(fcs_path).name}
from facs_autogating import FCSGatingPipeline
from facs_workflows_advanced import AdvancedGatingStrategies

# Chargement
pipeline = FCSGatingPipeline(
    '{fcs_path}',
    compensate=True,
    transform='logicle'
)

"""
    
    # Suggestions de gating
    suggestions = ChannelDetector.suggest_gating_strategy(detected)
    
    code += "# Séquence de gating suggérée:\n"
    for suggestion in suggestions:
        code += f"# {suggestion}\n"
    
    code += "\n# Code implémentable:\n"
    
    # Générer le code concret
    if 'FSC' in detected:
        code += """
# 1. Singlets
pipeline.gate_singlets_fsc_ssc(
    fsc_channel='FSC-A',
    fsc_h_channel='FSC-H'
)
"""
    
    if 'VIABILITY' in detected:
        viability_ch = detected['VIABILITY'][0]
        code += f"""
# 2. Cellules vivantes
AdvancedGatingStrategies.gate_live_dead(
    pipeline,
    viability_channel='{viability_ch}',
    parent_gate='singlets'
)
"""
    
    if 'T_CELLS' in detected:
        cd3_channels = [ch for ch in detected['T_CELLS'] if 'CD3' in ch.upper()]
        if cd3_channels:
            cd3_ch = cd3_channels[0]
            code += f"""
# 3. T cells (CD3+)
pipeline.gate_gmm_1d(
    '{cd3_ch}',
    n_components=2,
    select_component='positive',
    parent_gate='singlets'
)
"""
            
            cd4_channels = [ch for ch in detected['T_CELLS'] if 'CD4' in ch.upper()]
            cd8_channels = [ch for ch in detected['T_CELLS'] if 'CD8' in ch.upper()]
            
            if cd4_channels and cd8_channels:
                code += f"""
# 4. CD4/CD8 subsets
pipeline.gate_quadrants(
    '{cd4_channels[0]}',
    '{cd8_channels[0]}',
    parent_gate='singlets_{cd3_ch}_positive'
)
"""
    
    code += """
# Statistiques et export
stats = pipeline.compute_statistics()
pipeline.export_to_excel('results.xlsx')
print(stats[['Population', 'Count', 'Percentage_of_total']])
"""
    
    return code


if __name__ == "__main__":
    print("Module d'utilitaires FACS chargé")
    print("\nFonctionnalités disponibles:")
    print("  • FCSValidator: Validation de fichiers")
    print("  • ChannelDetector: Détection automatique de marqueurs")
    print("  • StatisticsExporter: Export avancé de statistiques")
    print("  • QualityControl: Contrôles qualité")
    print("  • auto_suggest_workflow: Génération automatique de workflow")
