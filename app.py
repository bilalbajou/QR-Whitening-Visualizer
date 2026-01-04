import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="QR Data Whitening", layout="wide")

st.title("Visualisation du Blanchiment de Données (Data Whitening) via QR")
st.markdown("""
Cette application démontre comment la décomposition QR peut être utilisée pour blanchir des données corrélées.
Le blanchiment transforme les données pour qu'elles aient une covariance d'identité (non corrélées et variance unitaire).
""")

# --- Sidebar Controls ---
st.sidebar.header("Paramètres")
n_samples = st.sidebar.slider("Nombre d'échantillons", 100, 1000, 500)
correlation = st.sidebar.slider("Niveau de corrélation (r)", -0.99, 0.99, 0.85)

# --- Data Generation ---
def generate_data(n, r):
    # Covariance matrix
    cov = [[1, r], [r, 1]]
    # Mean vector (0, 0)
    mean = [0, 0]
    # Generate data
    X = np.random.multivariate_normal(mean, cov, n)
    return X

X_original = generate_data(n_samples, correlation)

# Center the data explicitly (though multivariate_normal with mean 0 is statistically centered, sample mean might differ slightly)
X_centered = X_original - np.mean(X_original, axis=0)

# --- QR Decomposition & Whitening ---
# QR decomposition: X = Q * R
# X is shape (n, 2). Q is (n, 2) orthogonal. R is (2, 2) upper triangular.
# We want to transform X into Q (or scaled Q).
# Simple Whitening: X * R_inv = Q. 
# Since Q columns are orthonormal, Q^T Q = I.
# The covariance of Q is (Q^T Q) / (n-1) = I / (n-1).
# To get unit covariance, we typically scale by sqrt(n-1).
Q, R = np.linalg.qr(X_centered)
X_whitened = Q * np.sqrt(n_samples - 1)

# Ensure proper sign convention for visualization consistency (optional)
# Sometimes QR flips signs.

# --- Metrics ---
cov_original = np.cov(X_original, rowvar=False)
cov_whitened = np.cov(X_whitened, rowvar=False)

# --- Visualization ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Données Originales (Corrélées)")
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.scatter(X_original[:, 0], X_original[:, 1], alpha=0.5, color='blue')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_title(f"Corrélation: {cov_original[0, 1]:.2f}")
    ax1.grid(True)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    st.pyplot(fig1)
    
    st.write("**Matrice de Covariance Originale:**")
    st.code(np.array2string(cov_original, precision=3, suppress_small=True))

with col2:
    st.subheader("Données Blanchies (Décorrelées)")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    # Note: Use equal aspect ratio to show the "circle" shape correctly
    ax2.set_aspect('equal', adjustable='box')
    ax2.scatter(X_whitened[:, 0], X_whitened[:, 1], alpha=0.5, color='green')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_title(f"Corrélation: {cov_whitened[0, 1]:.2f}")
    ax2.grid(True)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    st.pyplot(fig2)

    st.write("**Matrice de Covariance Blanchie:**")
    st.code(np.array2string(cov_whitened, precision=3, suppress_small=True))

# --- Educational Text ---
st.divider()
st.header("Comment ça marche ?")
st.markdown(r"""
1. **Centrage** : On soustrait la moyenne pour centrer le nuage de points sur (0,0).
2. **Décomposition QR** : On décompose la matrice de données $X$ (centrée) en $X = Q R$.
   - $Q$ est une matrice orthogonale (ses colonnes sont orthogonales entre elles et de norme 1).
   - $R$ est une matrice triangulaire supérieure qui capture les relations linéaires (la corrélation) entre les variables.
3. **Blanchiment** : En multipliant $X$ par l'inverse de $R$ (ou simplement en prenant $Q$ remis à l'échelle), on supprime la distorsion causée par la corrélation.
   $$ X_{blanchi} = X R^{-1} = Q $$
   *(Avec un facteur d'échelle $\sqrt{n-1}$ pour rétablir la variance unitaire)*.
   
Ceci est très utile en Machine Learning (ex: avant une PCA ou des réseaux de neurones) pour que l'algorithme traite toutes les dimensions avec la même importance.
""")
