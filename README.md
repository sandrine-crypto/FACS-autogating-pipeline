# Pipeline d'Automatisation du Gating FACS
## Analyse automatisée de cytométrie en flux

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Pipeline complet et généralisable pour l'automatisation du gating en cytométrie en flux, inspiré des meilleures pratiques de FlowKit, OpenCyto et FlowWorkspace.

---

## 📋 Table des matières

- [Caractéristiques](#caractéristiques)
- [Installation](#installation)
- [Utilisation rapide](#utilisation-rapide)
- [Architecture](#architecture)
- [Méthodes de gating](#méthodes-de-gating)
- [Exemples avancés](#exemples-avancés)
- [Références scientifiques](#références-scientifiques)

---

## ✨ Caractéristiques

### Pipeline principal (`facs_autogating.py`)
- ✅ **Lecture de fichiers FCS** avec support de compensation et transformation
- ✅ **Gating automatisé** par multiple méthodes (GMM, DBSCAN, seuillage)
- ✅ **Stratégies hiérarchiques** de gating (singlets → viables → marqueurs)
- ✅ **Export Excel** avec statistiques détaillées et populations
- ✅ **Visualisations** de qualité publication
- ✅ **Architecture modulaire** pour cas complexes

### Workflows avancés (`facs_workflows_advanced.py`)
- 🔬 **Analyse par lot** (batch processing)
- 📊 **Comparaison multi-échantillons**
- 🧬 **Phénotypage immunitaire** avancé (T cells, B cells, NK, etc.)
- 📈 **Analyses statistiques** comparatives

---

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Installation des dépendances

```bash
# Cloner ou télécharger les fichiers
git clone https://github.com/votre-repo/facs-autogating.git
cd facs-autogating

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales
- **FlowKit** (≥1.1.0): Lecture et manipulation de fichiers FCS
- **scikit-learn** (≥1.3.0): Algorithmes de machine learning pour gating
- **pandas** (≥2.0.0): Manipulation de données tabulaires
- **matplotlib/seaborn**: Visualisations
- **openpyxl**: Export Excel

---

## 📖 Utilisation rapide

### Exemple basique: Un seul fichier

```python
from facs_autogating import FCSGatingPipeline

# 1. Charger le fichier FCS
pipeline = FCSGatingPipeline(
    'mon_echantillon.fcs',
    compensate=True,        # Appliquer compensation spectrale
    transform='logicle'     # Transformation logicle standard
)

# 2. Gating des singlets (cellules uniques)
pipeline.gate_singlets_fsc_ssc(
    fsc_channel='FSC-A',
    fsc_h_channel='FSC-H',
    method='linear_fit'
)

# 3. Suppression des débris
pipeline.gate_debris_removal(
    fsc_channel='FSC-A',
    ssc_channel='SSC-A',
    parent_gate='singlets'
)

# 4. Gating sur marqueur (ex: CD3+)
pipeline.gate_gmm_1d(
    channel='CD3-FITC',
    n_components=2,
    select_component='positive',
    parent_gate='singlets_viable'
)

# 5. Quadrants CD4/CD8
pipeline.gate_quadrants(
    'CD4-PE',
    'CD8-APC',
    parent_gate='singlets_viable_CD3-FITC_positive'
)

# 6. Statistiques
stats = pipeline.compute_statistics()
print(stats[['Population', 'Count', 'Percentage_of_total']])

# 7. Export
pipeline.export_to_excel('resultats.xlsx', include_populations=True)

# 8. Visualisation
pipeline.plot_gates('FSC-A', 'SSC-A', save_path='gates_FSC_SSC.png')
```

### Exemple workflow standard

```python
from facs_autogating import example_standard_workflow

# Exécution automatique d'un workflow complet
pipeline, stats = example_standard_workflow(
    fcs_path='mon_echantillon.fcs',
    output_dir='./resultats'
)
```

### Exemple analyse par lot

```python
from facs_workflows_advanced import BatchFCSAnalysis

# Liste de fichiers
fcs_files = ['control.fcs', 'treatment_A.fcs', 'treatment_B.fcs']
sample_names = ['Control', 'Treatment A', 'Treatment B']

# Initialisation
batch = BatchFCSAnalysis(fcs_files, sample_names)

# Analyse avec stratégie standard
pipelines = batch.run_standard_pipeline(
    compensate=True,
    transform='logicle',
    gate_strategy='standard'
)

# Comparaison entre échantillons
comparison = batch.compare_populations()
print(comparison)

# Export comparatif
batch.export_comparative_excel('analyse_comparative.xlsx')
batch.plot_comparative_barplot(save_path='comparaison.png')
```

---

## 🏗️ Architecture

### Structure du code

```
facs-autogating/
│
├── facs_autogating.py              # Module principal
│   ├── FCSGatingPipeline           # Classe principale pour gating
│   └── example_standard_workflow() # Workflow pré-configuré
│
├── facs_workflows_advanced.py      # Workflows avancés
│   ├── BatchFCSAnalysis            # Analyse par lot
│   └── AdvancedGatingStrategies    # Stratégies spécialisées
│
├── requirements.txt                # Dépendances
└── README.md                       # Documentation
```

### Classe `FCSGatingPipeline`

```python
class FCSGatingPipeline:
    """
    Pipeline modulaire pour gating automatisé
    
    Attributs principaux:
    - sample: Objet FlowKit Sample
    - data: DataFrame pandas avec les événements
    - gates: Dict {nom_gate: boolean_mask}
    - populations: Dict {nom_population: DataFrame}
    - stats: DataFrame avec statistiques
    """
    
    # Méthodes de gating
    def gate_singlets_fsc_ssc()      # Sélection singlets
    def gate_debris_removal()         # Suppression débris
    def gate_gmm_1d()                 # GMM 1D (biomodal)
    def gate_gmm_2d()                 # GMM 2D (multivarié)
    def gate_rectangle()              # Gating manuel rectangulaire
    def gate_quadrants()              # Quadrants (4 populations)
    
    # Analyse et export
    def compute_statistics()          # Calcul des stats
    def plot_gates()                  # Visualisations
    def export_to_excel()             # Export Excel complet
```

---

## 🔬 Méthodes de gating

### 1. Gating des singlets (doublet discrimination)

**Principe**: Exclure les doublets et agrégats cellulaires

**Méthode**: Régression linéaire robuste (RANSAC) entre FSC-A et FSC-H

```python
pipeline.gate_singlets_fsc_ssc(
    fsc_channel='FSC-A',      # Forward scatter area
    fsc_h_channel='FSC-H',    # Forward scatter height
    method='linear_fit',      # ou 'ratio'
    threshold=2.5             # Seuil en écarts-types
)
```

**Référence**: Shapiro (2003). *Practical Flow Cytometry*. 4th ed.

---

### 2. Suppression des débris (Debris removal)

**Principe**: Exclure débris cellulaires et petites particules

**Méthode**: Seuillage par percentiles sur FSC/SSC

```python
pipeline.gate_debris_removal(
    fsc_channel='FSC-A',
    ssc_channel='SSC-A',
    percentile_low=2,         # Percentile inférieur
    parent_gate='singlets'
)
```

---

### 3. Gaussian Mixture Models (GMM)

**Principe**: Modélisation de populations multimodales par mélange de gaussiennes

**Avantages**:
- Automatique (pas de seuil manuel)
- Adapté aux distributions bimodales (marqueurs +/-)
- Probabiliste

#### GMM 1D - Populations bimodales

```python
# Exemple: CD3+ vs CD3-
pipeline.gate_gmm_1d(
    channel='CD3-FITC',
    n_components=2,           # 2 gaussiennes (pos/neg)
    select_component='positive',  # Sélectionner composante haute
    parent_gate='singlets_viable'
)
```

**Référence**: Lo et al. (2008). *Cytometry A*. 73(4):321-332. "Automated gating of flow cytometry data via robust model-based clustering"

#### GMM 2D - Analyses multiparamétriques

```python
# Exemple: Analyse CD4 vs CD8
pipeline.gate_gmm_2d(
    channel_x='CD4-PE',
    channel_y='CD8-APC',
    n_components=4,           # 4 populations attendues
    select_components=[1, 2], # Indices des populations d'intérêt
    parent_gate='CD3_positive'
)
```

**Référence**: Aghaeepour et al. (2013). *Nat Methods*. 10(3):228-238. "Critical assessment of automated flow cytometry data analysis techniques"

---

### 4. Gating rectangulaire manuel

**Principe**: Définition manuelle de zones rectangulaires

```python
pipeline.gate_rectangle(
    channel_x='FSC-A',
    channel_y='SSC-A',
    x_min=30000, x_max=150000,
    y_min=0, y_max=100000,
    parent_gate='singlets'
)
```

---

### 5. Quadrants

**Principe**: Division en 4 quadrants (++, +-, -+, --)

```python
quadrants = pipeline.gate_quadrants(
    channel_x='CD4-PE',
    channel_y='CD8-APC',
    x_threshold=None,         # None = médiane automatique
    y_threshold=None,
    parent_gate='CD3_positive'
)

# Retourne:
# - CD4+CD8+ (double positifs)
# - CD4+CD8- (helpers)
# - CD4-CD8+ (cytotoxiques)
# - CD4-CD8- (double négatifs)
```

---

## 🧬 Exemples avancés

### Phénotypage T cells complet

```python
from facs_workflows_advanced import AdvancedGatingStrategies

# 1. Singlets et viabilité
pipeline.gate_singlets_fsc_ssc()
AdvancedGatingStrategies.gate_live_dead(
    pipeline,
    viability_channel='Live-Dead-APC-Cy7',
    parent_gate='singlets'
)

# 2. Leucocytes (CD45+ avec granularité)
AdvancedGatingStrategies.gate_cd45_ssc_leukocytes(
    pipeline,
    cd45_channel='CD45-V500',
    parent_gate='singlets_live_cells'
)
# Crée: lymphocytes, monocytes, granulocytes

# 3. T cells (CD3+)
pipeline.gate_gmm_1d('CD3-FITC', parent_gate='lymphocytes')

# 4. CD4+ vs CD8+
quadrants = pipeline.gate_quadrants(
    'CD4-PE',
    'CD8-PerCP',
    parent_gate='lymphocytes_CD3-FITC_positive'
)

# 5. Phénotype mémoire sur CD4+ (Naive, CM, EM, TEMRA)
AdvancedGatingStrategies.gate_memory_phenotype(
    pipeline,
    cd45ra_channel='CD45RA-APC',
    ccr7_channel='CCR7-BV421',
    parent_gate='lymphocytes_CD3-FITC_positive_CD4+CD8-'
)

# Résultats finaux
stats = pipeline.compute_statistics()
pipeline.export_to_excel('tcell_panel_complete.xlsx')
```

### Analyse comparative de traitement

```python
from facs_workflows_advanced import BatchFCSAnalysis

# Échantillons: contrôle et différentes doses de traitement
fcs_files = [
    'control_rep1.fcs', 'control_rep2.fcs', 'control_rep3.fcs',
    'drug_10uM_rep1.fcs', 'drug_10uM_rep2.fcs', 'drug_10uM_rep3.fcs',
    'drug_100uM_rep1.fcs', 'drug_100uM_rep2.fcs', 'drug_100uM_rep3.fcs'
]

sample_names = [
    'Control_1', 'Control_2', 'Control_3',
    '10µM_1', '10µM_2', '10µM_3',
    '100µM_1', '100µM_2', '100µM_3'
]

# Analyse par lot
batch = BatchFCSAnalysis(fcs_files, sample_names)
pipelines = batch.run_standard_pipeline(gate_strategy='lymphocytes')

# Comparaison statistique
comparison = batch.compare_populations()

# Populations d'intérêt
pops_of_interest = [
    'lymphocytes_CD3-FITC_positive_CD4+CD8-',  # CD4+ T cells
    'lymphocytes_CD3-FITC_positive_CD4-CD8+',  # CD8+ T cells
]

# Visualisation comparative
batch.plot_comparative_barplot(
    populations=pops_of_interest,
    save_path='drug_effect_on_tcells.png'
)

# Export détaillé
batch.export_comparative_excel('comparative_analysis_full.xlsx')
```

---

## 📊 Structure des exports Excel

### Fichier individuel (`pipeline.export_to_excel()`)

**Feuille 1: Statistics** - Statistiques complètes par population
- Population name
- Count (nombre d'événements)
- Percentage of total
- Mean/Median/Std pour chaque canal

**Feuille 2: File_Info** - Métadonnées du fichier
- Nom du fichier
- Nombre total d'événements
- Liste des canaux
- Paramètres de compensation/transformation
- Nombre de populations identifiées

**Feuille 3: Population_Counts** - Comptages simples
- Nom de population
- Count
- Percentage

**Feuilles 4+: Données brutes** (optionnel)
- Une feuille par population
- Tous les événements de la population
- Toutes les valeurs par canal

### Fichier comparatif (`batch.export_comparative_excel()`)

**Feuille 1: Overview** - Vue d'ensemble
- Sample name
- Total events
- Number of populations
- Number of channels

**Feuille 2: All_Statistics** - Toutes les stats
- Format long: Sample | Population | Count | % | Canal_mean | Canal_median | ...

**Feuille 3: Population_Counts** - Tableau pivot
- Populations en lignes
- Échantillons en colonnes
- Valeurs = counts

**Feuille 4: Population_Percentages** - Tableau pivot
- Populations en lignes
- Échantillons en colonnes
- Valeurs = pourcentages

---

## 🔧 Personnalisation

### Créer une stratégie de gating personnalisée

```python
def custom_gating_strategy(pipeline: FCSGatingPipeline):
    """Stratégie personnalisée pour panel spécifique"""
    
    # Étape 1: QC standard
    pipeline.gate_singlets_fsc_ssc()
    pipeline.gate_debris_removal(parent_gate='singlets')
    
    # Étape 2: Votre logique spécifique
    # Exemple: Panel NK cells
    
    # CD3- (exclusion T cells)
    pipeline.gate_gmm_1d(
        'CD3-FITC',
        n_components=2,
        select_component='negative',
        parent_gate='singlets_viable'
    )
    
    # CD56+ CD16+ (NK cells)
    pipeline.gate_gmm_2d(
        'CD56-PE',
        'CD16-APC',
        n_components=3,
        parent_gate='singlets_viable_CD3-FITC_negative'
    )
    
    # Sous-populations NK
    pipeline.gate_quadrants(
        'CD56-PE',
        'CD16-APC',
        parent_gate='singlets_viable_CD3-FITC_negative'
    )
    
    return pipeline

# Utilisation
pipeline = FCSGatingPipeline('nk_panel.fcs')
pipeline = custom_gating_strategy(pipeline)
stats = pipeline.compute_statistics()
```

---

## 📚 Références scientifiques

### Fondamentaux de la cytométrie en flux

1. **Shapiro HM** (2003). *Practical Flow Cytometry*. 4th ed. Wiley-Liss. ISBN: 978-0471411253
   - Référence classique pour la théorie et pratique de la cytométrie

2. **Herzenberg LA et al.** (2006). *Nat Immunol*. 7(7):681-685. "Interpreting flow cytometry data: a guide for the perplexed"
   - Guide pour l'interprétation des données

### Gating automatisé et algorithmes

3. **Lo K et al.** (2008). *Cytometry A*. 73(4):321-332. "Automated gating of flow cytometry data via robust model-based clustering"
   - Base théorique pour GMM en cytométrie

4. **Aghaeepour N et al.** (2013). *Nat Methods*. 10(3):228-238. "Critical assessment of automated flow cytometry data analysis techniques"
   - Comparaison exhaustive des méthodes d'analyse automatisée

5. **Finak G et al.** (2014). *Bioinformatics*. 30(9):1274-1281. "OpenCyto: an open source infrastructure for scalable, robust, reproducible, and automated, end-to-end flow cytometry data analysis"
   - Infrastructure OpenCyto (R/Bioconductor)

6. **Spidlen J et al.** (2021). *Cytometry A*. 99(1):100-102. "FlowRepository: A resource of annotated flow cytometry datasets associated with peer-reviewed publications"
   - Base de données publiques pour benchmarking

### Transformations et normalisation

7. **Parks DR et al.** (2006). *Cytometry A*. 69(6):541-551. "A new 'Logicle' display method avoids deceptive effects of logarithmic scaling for low signals and compensated data"
   - Transformation Logicle standard

8. **Bagwell CB** (2005). *Cytometry A*. 68(1):36-43. "Hyperlog-a flexible log-like transform for negative, zero, and positive valued data"
   - Transformation Hyperlog alternative

### Standards et bonnes pratiques

9. **Maecker HT et al.** (2005). *Nat Rev Immunol*. 5(9):699-710. "Standardizing immunophenotyping for the Human Immunology Project"
   - Standardisation des panels d'immunophénotypage

10. **Brinkman RR et al.** (2016). *Cytometry A*. 89(4):292-294. "MIFC: Minimal Information for Flow Cytometry"
    - Standards de reporting pour publications

---

## 🐛 Dépannage

### Problème: "FileNotFoundError" lors du chargement FCS

**Solution**: Vérifier le chemin absolu du fichier
```python
from pathlib import Path
fcs_path = Path('mon_fichier.fcs').resolve()
pipeline = FCSGatingPipeline(str(fcs_path))
```

### Problème: Transformation "logicle" échoue

**Solution**: Tester transformation alternative ou désactiver
```python
pipeline = FCSGatingPipeline(
    'fichier.fcs',
    transform='asinh'  # ou None pour pas de transformation
)
```

### Problème: GMM ne trouve pas les bonnes populations

**Solutions**:
1. Vérifier la transformation des données
2. Augmenter/diminuer le nombre de composantes
3. Utiliser gating manuel comme alternative

```python
# Au lieu de GMM automatique
pipeline.gate_gmm_1d('CD3', n_components=2)

# Utiliser seuil manuel
threshold = pipeline.data['CD3'].median() * 1.5
gate = pipeline.data['CD3'] > threshold
```

### Problème: Fichier Excel trop volumineux

**Solution**: Désactiver l'export des populations brutes
```python
pipeline.export_to_excel(
    'resultats.xlsx',
    include_populations=False  # Ne pas inclure les données brutes
)
```

---

## 📄 License

MIT License - Libre d'utilisation pour recherche académique et applications commerciales

---

## 🤝 Contributions

Les contributions sont bienvenues! Merci de:
1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## 📧 Contact

Pour questions, suggestions ou collaborations: [votre-email]

---

## 🙏 Remerciements

Inspiré par les excellents packages:
- **FlowKit** (Python): https://github.com/whitews/FlowKit
- **OpenCyto** (R/Bioconductor): https://github.com/RGLab/openCyto
- **flowWorkspace** (R/Bioconductor): https://github.com/RGLab/flowWorkspace
- **FlowJo**: https://www.flowjo.com/ (commercial, inspiration UI/UX)

---

*Dernière mise à jour: Décembre 2024*
