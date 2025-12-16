# 🚀 Guide d'Installation GitHub et Déploiement Streamlit

## Guide Complet Pas à Pas

---

## 📋 Table des Matières

1. [Prérequis](#prérequis)
2. [Installation Locale](#installation-locale)
3. [Configuration GitHub](#configuration-github)
4. [Déploiement Streamlit Cloud](#déploiement-streamlit-cloud)
5. [Utilisation](#utilisation)
6. [Dépannage](#dépannage)

---

## 1️⃣ Prérequis

### Logiciels requis

- **Python 3.8+** : [Télécharger](https://www.python.org/downloads/)
- **Git** : [Télécharger](https://git-scm.com/downloads)
- **Compte GitHub** : [Créer un compte](https://github.com/signup)
- **Compte Streamlit Cloud** : [Créer un compte](https://streamlit.io/cloud) (gratuit)

### Vérifier les installations

```bash
# Vérifier Python
python --version
# ou
python3 --version

# Vérifier Git
git --version

# Vérifier pip
pip --version
```

---

## 2️⃣ Installation Locale

### Étape 1 : Télécharger les fichiers

Récupérez tous les fichiers du pipeline FACS dans un dossier local, par exemple :

```bash
mkdir facs-autogating
cd facs-autogating
```

### Étape 2 : Créer un environnement virtuel

**Sur Windows :**
```bash
python -m venv venv
venv\Scripts\activate
```

**Sur macOS/Linux :**
```bash
python3 -m venv venv
source venv/bin/activate
```

Vous devriez voir `(venv)` apparaître dans votre terminal.

### Étape 3 : Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 4 : Tester l'application localement

```bash
streamlit run streamlit_app.py
```

Votre navigateur devrait s'ouvrir automatiquement à `http://localhost:8501`

---

## 3️⃣ Configuration GitHub

### Étape 1 : Créer un dépôt GitHub

1. **Aller sur GitHub** : https://github.com
2. **Cliquer sur "New repository"** (bouton vert en haut à droite)
3. **Remplir les informations** :
   - Repository name : `facs-autogating-pipeline`
   - Description : `Pipeline d'automatisation du gating pour cytométrie en flux`
   - Visibilité : Public (ou Private selon vos besoins)
   - ✅ Cocher "Add a README file"
4. **Cliquer sur "Create repository"**

### Étape 2 : Cloner le dépôt localement

```bash
# Remplacer VOTRE-USERNAME par votre nom d'utilisateur GitHub
git clone https://github.com/VOTRE-USERNAME/facs-autogating-pipeline.git
cd facs-autogating-pipeline
```

### Étape 3 : Copier les fichiers du pipeline

Copier tous les fichiers du pipeline dans ce dossier :

```
facs-autogating-pipeline/
├── .gitignore
├── README.md
├── requirements.txt
├── streamlit_app.py
├── facs_autogating.py
├── facs_workflows_advanced.py
├── facs_utilities.py
├── facs_cli.py
├── FACS_Tutorial.ipynb
└── REFERENCES_BIBLIOGRAPHIQUES.md
```

### Étape 4 : Configuration Git (première fois seulement)

```bash
# Configurer votre identité
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"
```

### Étape 5 : Ajouter et commiter les fichiers

```bash
# Ajouter tous les fichiers
git add .

# Vérifier les fichiers ajoutés
git status

# Commiter avec un message
git commit -m "Premier commit : Pipeline FACS complet avec interface Streamlit"

# Pousser vers GitHub
git push origin main
```

**Note** : Si vous obtenez une erreur "main doesn't exist", essayez :
```bash
git push origin master
```

### Étape 6 : Vérifier sur GitHub

1. Retourner sur votre dépôt GitHub : `https://github.com/VOTRE-USERNAME/facs-autogating-pipeline`
2. Vérifier que tous les fichiers sont présents

---

## 4️⃣ Déploiement Streamlit Cloud

### Étape 1 : Créer un compte Streamlit Cloud

1. Aller sur : https://streamlit.io/cloud
2. Cliquer sur "Sign up" ou "Get started"
3. **Se connecter avec GitHub** (recommandé)
4. Autoriser Streamlit à accéder à vos dépôts

### Étape 2 : Déployer l'application

1. **Une fois connecté**, cliquer sur "New app"
2. **Remplir les informations** :
   - Repository : Sélectionner `VOTRE-USERNAME/facs-autogating-pipeline`
   - Branch : `main` (ou `master`)
   - Main file path : `streamlit_app.py`
   - App URL (optionnel) : Personnaliser l'URL
3. **Cliquer sur "Deploy!"**

### Étape 3 : Attendre le déploiement

- Le déploiement prend généralement 2-5 minutes
- Vous verrez les logs de déploiement en temps réel
- Une fois terminé, l'application sera accessible via l'URL fournie

### Étape 4 : Obtenir l'URL de votre application

Format de l'URL : `https://VOTRE-APP-NAME.streamlit.app`

Exemple : `https://facs-autogating.streamlit.app`

---

## 5️⃣ Utilisation

### Interface Streamlit

Une fois déployée, votre application est accessible publiquement via l'URL Streamlit Cloud.

#### Mode 1 : Analyse Simple
1. Sélectionner "🔍 Analyse Simple" dans la barre latérale
2. Télécharger un fichier FCS
3. Configurer les options de gating
4. Cliquer sur "Lancer l'Analyse"
5. Explorer les résultats dans les onglets

#### Mode 2 : Analyse par Lot
1. Sélectionner "📊 Analyse par Lot"
2. Télécharger plusieurs fichiers FCS
3. Choisir la stratégie de gating
4. Lancer l'analyse comparative

#### Mode 3 : Validation
1. Sélectionner "✅ Validation de Fichiers"
2. Télécharger des fichiers à valider
3. Voir le rapport de validation

#### Mode 4 : Détection Automatique
1. Sélectionner "🎯 Détection Automatique"
2. Télécharger un fichier FCS
3. Obtenir les suggestions de workflow
4. Télécharger le code Python généré

### Utilisation en Ligne de Commande (Local)

```bash
# Analyser un fichier
python facs_cli.py analyze echantillon.fcs -o ./resultats

# Analyse par lot
python facs_cli.py batch -l file_list.txt -o ./resultats

# Valider des fichiers
python facs_cli.py validate echantillon.fcs

# Suggérer un workflow
python facs_cli.py suggest echantillon.fcs -o workflow.py

# Lister les canaux
python facs_cli.py channels echantillon.fcs
```

### Utilisation en Python (Local)

```python
from facs_autogating import FCSGatingPipeline

# Charger et analyser
pipeline = FCSGatingPipeline('echantillon.fcs', compensate=True, transform='logicle')
pipeline.gate_singlets_fsc_ssc()
pipeline.gate_debris_removal(parent_gate='singlets')

# Statistiques
stats = pipeline.compute_statistics()
print(stats[['Population', 'Count', 'Percentage_of_total']])

# Export
pipeline.export_to_excel('resultats.xlsx')
```

---

## 6️⃣ Mise à Jour du Code

### Modifier le code localement

```bash
# Faire vos modifications dans les fichiers Python

# Vérifier les modifications
git status

# Ajouter les fichiers modifiés
git add .

# Commiter
git commit -m "Description des modifications"

# Pousser vers GitHub
git push origin main
```

### Déploiement automatique

- **Streamlit Cloud redéploie automatiquement** lorsque vous poussez des modifications sur GitHub
- Le redéploiement prend ~2 minutes
- Vous pouvez suivre le processus dans le dashboard Streamlit Cloud

---

## 7️⃣ Configuration Avancée

### Fichier de configuration Streamlit (optionnel)

Créer un fichier `.streamlit/config.toml` pour personnaliser l'interface :

```bash
mkdir .streamlit
```

Créer le fichier `config.toml` :

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 500
enableXsrfProtection = true
```

### Secrets pour données sensibles (si nécessaire)

Si vous avez besoin de stocker des clés API ou mots de passe :

1. Dans Streamlit Cloud : Settings > Secrets
2. Ajouter vos secrets au format TOML :

```toml
api_key = "votre_cle_api"
database_password = "votre_mot_de_passe"
```

3. Accéder dans le code :

```python
import streamlit as st
api_key = st.secrets["api_key"]
```

---

## 8️⃣ Dépannage

### Problème : "Module not found"

**Solution** : Vérifier que `requirements.txt` contient toutes les dépendances

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Mise à jour des dépendances"
git push origin main
```

### Problème : Application ne démarre pas

**Solution** : Vérifier les logs dans Streamlit Cloud
- Aller dans "Manage app" > "Logs"
- Identifier l'erreur
- Corriger le code localement
- Pousser les modifications

### Problème : Fichiers trop volumineux

**Solution** : GitHub a une limite de 100 MB par fichier

- Ne jamais commiter de fichiers `.fcs` ou données brutes
- Vérifier que `.gitignore` est bien configuré
- Si un gros fichier est déjà committé :

```bash
# Supprimer de l'historique (attention, opération avancée)
git filter-branch --tree-filter 'rm -rf data/' HEAD
git push origin main --force
```

### Problème : Limite de mémoire Streamlit Cloud

**Solution** : Streamlit Cloud gratuit a des limites de ressources
- Optimiser le code pour réduire l'utilisation mémoire
- Traiter les fichiers un par un plutôt qu'en batch
- Envisager Streamlit Cloud Community (payant) pour plus de ressources

### Problème : L'application est lente

**Solutions** :
1. Utiliser le cache Streamlit :

```python
@st.cache_data
def load_data(file_path):
    # Votre code
    return data
```

2. Optimiser les visualisations (réduire le nombre de points affichés)
3. Utiliser `rasterized=True` dans matplotlib

---

## 9️⃣ Commandes Git Utiles

```bash
# Voir l'état des fichiers
git status

# Voir l'historique des commits
git log

# Créer une nouvelle branche
git checkout -b nouvelle-feature

# Changer de branche
git checkout main

# Fusionner une branche
git merge nouvelle-feature

# Annuler les modifications non commitées
git checkout -- fichier.py

# Voir les différences
git diff

# Récupérer les dernières modifications de GitHub
git pull origin main
```

---

## 🔟 Partage et Collaboration

### Rendre le dépôt public

1. Aller dans Settings du dépôt GitHub
2. Scroll vers le bas jusqu'à "Danger Zone"
3. Cliquer sur "Change visibility"
4. Choisir "Public"

### Inviter des collaborateurs

1. Aller dans Settings > Collaborators
2. Cliquer sur "Add people"
3. Entrer le nom d'utilisateur GitHub
4. Choisir les permissions (Read, Write, Admin)

### Créer une Release

1. Aller dans l'onglet "Releases"
2. Cliquer sur "Create a new release"
3. Tag : `v1.0.0`
4. Titre : "Version 1.0 - Initial Release"
5. Description : Liste des fonctionnalités
6. Cliquer sur "Publish release"

---

## 📊 Monitoring et Analytics

### Streamlit Cloud Analytics

- Dashboard Streamlit Cloud montre :
  - Nombre de visiteurs
  - Temps de chargement
  - Erreurs
  - Utilisation des ressources

### GitHub Insights

- Onglet "Insights" sur GitHub montre :
  - Activité du dépôt
  - Contributeurs
  - Trafic
  - Clones

---

## 🎓 Ressources Supplémentaires

### Documentation

- **Streamlit** : https://docs.streamlit.io
- **Git** : https://git-scm.com/doc
- **GitHub** : https://docs.github.com
- **Python** : https://docs.python.org/3/

### Tutoriels

- [Git for Beginners](https://www.freecodecamp.org/news/git-for-beginners/)
- [Streamlit Tutorial](https://docs.streamlit.io/get-started)
- [GitHub Actions](https://docs.github.com/en/actions)

---

## 📧 Support

Pour toute question ou problème :
1. Vérifier les [Issues GitHub](https://github.com/VOTRE-USERNAME/facs-autogating-pipeline/issues)
2. Créer une nouvelle Issue si nécessaire
3. Consulter la documentation

---

## ✅ Checklist de Déploiement

- [ ] Python 3.8+ installé
- [ ] Git installé et configuré
- [ ] Compte GitHub créé
- [ ] Dépôt GitHub créé
- [ ] Tous les fichiers poussés sur GitHub
- [ ] Compte Streamlit Cloud créé
- [ ] Application déployée sur Streamlit Cloud
- [ ] Application accessible via URL
- [ ] Tests effectués sur l'application déployée
- [ ] README mis à jour avec l'URL de l'application
- [ ] `.gitignore` configuré correctement

---

**🎉 Félicitations ! Votre pipeline FACS est maintenant en ligne et accessible à tous !**

---

*Dernière mise à jour : Décembre 2024*
