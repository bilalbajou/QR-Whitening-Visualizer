# Visualiseur de Blanchiment de Données via QR (QR-Whitening Visualizer)

Une application interactive basée sur Python et Streamlit pour démontrer le concept de blanchiment de données (Data Whitening) en utilisant la décomposition QR.

## 📋 Description

Ce projet a pour but pédagogique d'illustrer comment des données corrélées peuvent être transformées en données non corrélées (sphériques) avec une variance unitaire. Cette technique est souvent utilisée comme étape de prétaitement en Machine Learning pour améliorer la convergence des algorithmes.

L'application permet de :
1. Générer des données synthétiques avec un niveau de corrélation ajustable.
2. Visualiser ces données sous forme de nuage de points (forme elliptique).
3. Appliquer une décomposition QR pour "blanchir" les données.
4. Visualiser le résultat transformé (forme circulaire) et vérifier les matrices de covariance.

## 🛠️ Prérequis

- Python 3.8 ou supérieur
- Pip (gestionnaire de paquets Python)

## 📦 Installation

1. Clonez ce dépôt ou téléchargez les fichiers dans un dossier local.
2. Ouvrez un terminal dans le dossier du projet (`d:\vibe coding project\QR-Whitening Visualizer`).
3. Installez les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

### Dépendances principales
- `streamlit` : Pour l'interface web interactive.
- `numpy` : Pour les calculs matriciels et la génération de données.
- `matplotlib` : Pour la visualisation graphique.
- `seaborn` : Pour l'esthétique des graphiques.

## 🚀 Utilisation

Pour lancer l'application, exécutez la commande suivante dans votre terminal :

```bash
streamlit run app.py
```

Une fois lancée, l'application s'ouvrira automatiquement dans votre navigateur par défaut (généralement à l'adresse `http://localhost:8501`).

## 🧮 Concept Mathématique : Décomposition QR

Le blanchiment des données vise à transformer un vecteur aléatoire $X$ (centré) d'une covariance $\Sigma$ quelconque en un vecteur $X_{blanchi}$ de covariance identité $I$.

Dans cette application, nous utilisons la **décomposition QR** de la matrice de données $X$ (de taille $n \times d$) :

$$ X = Q R $$

Où :
- $Q$ est une matrice orthogonale ($n \times d$) telle que $Q^T Q = I$ (à un facteur d'échelle près selon la convention).
- $R$ est une matrice triangulaire supérieure ($d \times d$).

En multipliant $X$ par $R^{-1}$ :

$$ X R^{-1} = Q $$

Les colonnes de $Q$ sont orthogonales, ce qui signifie que les nouvelles variables sont décorrélées. En ajustant l'échelle par $\sqrt{n-1}$, on obtient une variance unitaire.

## 📂 Structure du Projet

```
QR-Whitening Visualizer/
├── app.py              # Code principal de l'application Streamlit
├── requirements.txt    # Liste des librairies requises
└── README.md           # Documentation du projet
```

