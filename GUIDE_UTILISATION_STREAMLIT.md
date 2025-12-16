# 🌐 Interface Web Streamlit - Guide d'Utilisation

## Interface Utilisateur Conviviale pour le Pipeline FACS

---

## 🎯 Accès à l'Application

### En ligne (Streamlit Cloud)
**URL** : `https://votre-app.streamlit.app`

### En local
```bash
streamlit run streamlit_app.py
```
L'application s'ouvrira automatiquement dans votre navigateur à `http://localhost:8501`

---

## 📱 Navigation

L'interface comprend **4 modes principaux** accessibles via la barre latérale :

### 🔍 Mode 1 : Analyse Simple
**Pour** : Analyser un seul fichier FCS

**Étapes** :
1. Télécharger un fichier FCS
2. Configurer les paramètres de gating :
   - ✅ Gate Singlets
   - ✅ Supprimer les débris
   - ✅ Gating automatique des marqueurs
   - ☐ Créer quadrants CD4/CD8
3. Cliquer sur "🚀 Lancer l'Analyse"
4. Explorer les résultats dans les onglets :
   - **📈 Statistiques** : Tableaux et métriques
   - **🎨 Visualisations** : Scatter plots interactifs
   - **📋 Détails** : Informations complètes
   - **💾 Export** : Télécharger Excel et CSV

**Temps estimé** : 30 secondes - 2 minutes selon la taille du fichier

---

### 📊 Mode 2 : Analyse par Lot
**Pour** : Comparer plusieurs échantillons

**Étapes** :
1. Télécharger plusieurs fichiers FCS (glisser-déposer)
2. Choisir la stratégie de gating :
   - `standard` : QC basique (singlets + débris)
   - `lymphocytes` : Panel T cells complet
3. Cocher "Générer graphiques comparatifs"
4. Lancer l'analyse
5. Consulter les résultats comparatifs :
   - **Vue d'ensemble** : Tableaux comparatifs
   - **Graphiques** : Barres groupées
   - **Export** : Excel comparatif

**Temps estimé** : 1-5 minutes selon le nombre de fichiers

---

### ✅ Mode 3 : Validation de Fichiers
**Pour** : Vérifier la qualité de vos fichiers FCS

**Étapes** :
1. Télécharger un ou plusieurs fichiers FCS
2. Cliquer sur "🔍 Valider les Fichiers"
3. Consulter le rapport de validation :
   - ✅/❌ Validité
   - Nombre d'événements
   - Canaux disponibles
   - Matrice de compensation
   - Avertissements éventuels

**Utilité** : Avant toute analyse, s'assurer de la qualité des données

---

### 🎯 Mode 4 : Détection Automatique
**Pour** : Obtenir une suggestion de workflow adaptée à votre panel

**Étapes** :
1. Télécharger un fichier FCS
2. Cliquer sur "🔍 Analyser et Suggérer Workflow"
3. Consulter :
   - **Marqueurs détectés** : CD3, CD4, CD8, etc.
   - **Workflow suggéré** : Étapes de gating recommandées
   - **Code Python généré** : Script prêt à l'emploi
4. Télécharger le code Python

**Utilité** : Idéal pour nouveaux panels ou utilisateurs débutants

---

## ⚙️ Paramètres Globaux

Dans la **barre latérale**, vous pouvez configurer :

### Compensation Spectrale
- ✅ **Activée** : Applique la matrice de compensation du fichier FCS
- ☐ Désactivée : Utilise les données brutes

**Recommandation** : Toujours activée sauf si déjà appliquée lors de l'acquisition

### Transformation des Données
Options disponibles :
- **logicle** (recommandé) : Standard pour données compensées
- **asinh** : Alternative pour données négatives
- **hyperlog** : Similaire à logicle
- **aucune** : Données linéaires

**Recommandation** : `logicle` pour la plupart des cas

---

## 📊 Comprendre les Résultats

### Statistiques des Populations

**Colonnes principales** :
- **Population** : Nom du gate (ex: "singlets", "CD3_positive")
- **Nombre** : Nombre d'événements dans cette population
- **% du Total** : Pourcentage par rapport au total d'événements

### Visualisations

**Scatter Plots** :
- Points gris : Événements exclus
- Points rouges : Population sélectionnée
- Axes : Canaux sélectionnés (FSC, SSC, marqueurs)

**Personnalisation** :
- Choisir les canaux X et Y
- Sélectionner les populations à afficher
- Télécharger les figures en haute résolution

### Exports

**Fichier Excel** contient :
- Feuille 1 : Statistiques complètes
- Feuille 2 : Informations du fichier
- Feuille 3 : Comptages des populations
- Feuilles suivantes : Données brutes par population (optionnel)

**Fichier CSV** :
- Format simple pour analyses ultérieures
- Compatible Excel, R, Python

---

## 🔧 Cas d'Usage Typiques

### Cas 1 : Immunophénotypage PBMC Standard

**Objectif** : Quantifier les lymphocytes T CD4+ et CD8+

**Workflow** :
1. Mode "Analyse Simple"
2. Télécharger le fichier PBMC
3. Activer tous les gates (singlets, débris, marqueurs)
4. L'application détectera automatiquement CD3, CD4, CD8
5. Les quadrants CD4/CD8 seront créés
6. Télécharger le rapport Excel

**Résultat attendu** :
- Population CD4+ : ~40-60% des CD3+
- Population CD8+ : ~20-40% des CD3+

---

### Cas 2 : Comparaison Avant/Après Traitement

**Objectif** : Comparer l'effet d'un traitement sur les populations cellulaires

**Workflow** :
1. Mode "Analyse par Lot"
2. Télécharger :
   - Échantillons contrôle (n=3)
   - Échantillons traités (n=3)
3. Choisir stratégie "lymphocytes"
4. Lancer l'analyse comparative
5. Consulter le graphique en barres
6. Télécharger l'Excel comparatif

**Analyses possibles** :
- Évolution des populations CD4+ et CD8+
- Changements dans les sous-populations mémoire
- Activation cellulaire (CD69+, CD25+)

---

### Cas 3 : Validation de Qualité

**Objectif** : Vérifier la qualité avant analyse

**Workflow** :
1. Mode "Validation"
2. Télécharger tous les fichiers d'une expérience
3. Vérifier :
   - ✅ Tous les fichiers valides
   - ≥10,000 événements par fichier
   - Compensation présente
   - Pas d'avertissements critiques

**Action si problèmes** :
- Fichiers invalides : Réacquérir
- Peu d'événements : Augmenter temps d'acquisition
- Pas de compensation : Appliquer avant export

---

## 💡 Conseils et Bonnes Pratiques

### Préparation des Fichiers

✅ **À FAIRE** :
- Nommer les fichiers de façon claire (ex: `Ctrl_Rep1.fcs`, `Drug_10uM_Rep2.fcs`)
- Appliquer la compensation pendant l'acquisition (si possible)
- Enregistrer en format FCS 3.0 ou 3.1
- Inclure au moins 10,000 événements par fichier

❌ **À ÉVITER** :
- Noms de fichiers avec caractères spéciaux (#, @, %, espaces)
- Fichiers trop volumineux (>100 MB) - fragmenter l'acquisition
- Mélange de panels différents dans une analyse par lot

### Optimisation des Analyses

**Pour fichiers volumineux (>500,000 événements)** :
- Utiliser le sous-échantillonnage dans le cytomètre
- Ou analyser par lot de 100,000-200,000 événements

**Pour analyses répétitives** :
- Utiliser le mode "Détection Automatique" une fois
- Sauvegarder le code Python généré
- Réutiliser le script en ligne de commande

### Interprétation des Résultats

**Contrôles de Qualité** :
- Doublets exclus : Devrait être <10%
- Débris exclus : Variable selon le type cellulaire
- Populations négatives : Vérifier avec FMO (Fluorescence Minus One)

**Populations attendues (PBMC humain)** :
- Lymphocytes : 60-90% des leucocytes
- CD3+ T cells : 60-80% des lymphocytes
- CD4+ : 40-60% des CD3+
- CD8+ : 20-40% des CD3+
- CD19+ B cells : 5-20% des lymphocytes
- CD56+ NK : 5-15% des lymphocytes

---

## 🚨 Dépannage

### Problème : "Fichier trop volumineux"

**Cause** : Limite Streamlit Cloud (500 MB)

**Solutions** :
1. Compresser le fichier FCS
2. Réduire le nombre d'événements à l'acquisition
3. Utiliser la version locale

### Problème : "Analyse très lente"

**Causes** :
- Fichier trop volumineux
- Trop de visualisations simultanées
- Connexion Internet lente (si cloud)

**Solutions** :
1. Désélectionner certaines populations dans les visualisations
2. Télécharger moins de fichiers en lot
3. Utiliser la version locale pour fichiers >200,000 événements

### Problème : "Marqueurs non détectés"

**Cause** : Nomenclature non standard des canaux

**Solution** :
- Vérifier les noms dans "Mode Détection Automatique"
- Utiliser les noms exacts dans l'interface
- Renommer les canaux dans le logiciel d'acquisition

### Problème : "Export Excel échoue"

**Cause** : Trop de populations ou données trop volumineuses

**Solution** :
- Décocher "Inclure populations" dans l'export
- Exporter seulement les statistiques (CSV)
- Utiliser la version ligne de commande pour export personnalisé

---

## 📈 Fonctionnalités Avancées

### Cache et Performance

L'application utilise le cache Streamlit pour :
- Éviter de recharger les mêmes fichiers
- Accélérer les visualisations
- Réduire la latence

**Rafraîchir le cache** : Recharger la page (F5)

### Export Programmatique

Pour automatiser les exports, utiliser l'API Python directement :

```python
from facs_autogating import FCSGatingPipeline

pipeline = FCSGatingPipeline('fichier.fcs')
pipeline.gate_singlets_fcs_ssc()
# ... autres gates
pipeline.export_to_excel('resultats.xlsx')
```

Voir `facs_cli.py` pour ligne de commande

---

## 📞 Support et Feedback

### Problèmes Techniques
- Créer une Issue sur GitHub : [Lien vers Issues]
- Vérifier la documentation : README.md

### Suggestions de Fonctionnalités
- Utiliser GitHub Discussions
- Proposer une Pull Request

### Questions Scientifiques
- Consulter REFERENCES_BIBLIOGRAPHIQUES.md
- Contacter le mainteneur du projet

---

## 🎓 Tutoriels Vidéo (à venir)

- [ ] Analyse simple d'un échantillon PBMC
- [ ] Analyse comparative de traitement
- [ ] Personnalisation des stratégies de gating
- [ ] Export et post-traitement dans R/Python

---

## ✅ Checklist Utilisateur

Avant chaque analyse :
- [ ] Fichiers FCS valides et <100 MB
- [ ] Nomenclature des canaux vérifiée
- [ ] Compensation appliquée
- [ ] Contrôles appropriés disponibles
- [ ] Stratégie de gating planifiée

Après chaque analyse :
- [ ] Résultats cohérents biologiquement
- [ ] Statistiques exportées
- [ ] Visualisations sauvegardées
- [ ] Fichiers nommés et organisés

---

**🎉 Bon gating !**

---

*Version 1.0 - Décembre 2024*
