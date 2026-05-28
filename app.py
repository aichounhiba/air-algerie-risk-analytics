import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Configuration de la page et application du Thème "Rouge & Blanc" Air Algérie
st.set_page_config(
    page_title="Air Algérie - Risk Analytics",
    page_icon="✈️",
    layout="wide"
)

# Injection de CSS pour forcer le thème aux couleurs d'Air Algérie
st.markdown("""
    <style>
        /* Couleur de fond principale et texte */
        .stApp {
            background-color: #FFFFFF;
            color: #333333;
        }
        /* Personnalisation des titres */
        h1, h2, h3 {
            color: #D2143A !important; /* Rouge Air Algérie */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Personnalisation des boutons et widgets */
        div.stButton > button:first-child {
            background-color: #D2143A;
            color: white;
            border-radius: 5px;
            border: none;
        }
        div.stButton > button:first-child:hover {
            background-color: #B00F2E;
            color: white;
        }
        /* Cartes de métriques */
        [data-testid="stMetricValue"] {
            color: #D2143A !important;
        }
    </style>
""", unsafe_allow_html=True)

# 2. En-tête de l'application
st.title("✈️ Air Algérie - Risk & Financial Analytics Dashboard")
st.subheader("Optimisation du Risque Carburant et Gestion Actuarielle des Retards")
st.markdown("---")

# 3. Barre latérale pour les paramètres (Sidebar)
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/23/Air_Alg%C3%A9rie_logo.svg", width=150) # Logo indicatif
st.sidebar.header("⚙️ Paramètres de Simulation")

# Paramètres Carburant
st.sidebar.subheader("⛽ Risque Carburant (Finance)")
prix_actuel = st.sidebar.slider("Prix actuel du baril (USD)", 60, 150, 85)
volatilite = st.sidebar.slider("Volatilité annuelle du marché (%)", 10, 50, 25) / 100
taux_couverture = st.sidebar.slider("Taux de couverture souhaité (Hedging %)", 0, 100, 60) / 100

# Paramètres Actuariat Retards
st.sidebar.subheader("⏱️ Risque Opérationnel (Actuariat)")
lambda_poisson = st.sidebar.slider("Fréquence moyenne des retards (Loi de Poisson / mois)", 10, 100, 45)
cout_moyen = st.sidebar.slider("Coût moyen d'indemnisation par retard (DA)", 50000, 300000, 120000, step=10000)

# 4. Corps principal : Division en onglets
tab1, tab2 = st.tabs(["📊 Couverture Kérosène (Hedging)", "🧮 Modélisation Actuarielle des Retards"])

# --- ONGLET 1 : FINANCE (RISQUE CARBURANT) ---
with tab1:
    st.header("Gestion du Risque de Volatilité du Kérosène")
    st.write("Simulation de l'évolution du prix du carburant sur 12 mois avec et sans stratégie de couverture financière.")
    
    # Simulation d'une trajectoire (Mouvement Brownien Géométrique)
    np.random.seed(42)
    mois = np.arange(1, 13)
    rendements = np.random.normal(0.005, volatilite / np.sqrt(12), 12)
    prix_sans_couverture = prix_actuel * np.exp(np.cumsum(rendements))
    
    # Prix avec couverture (Hedging)
    prix_fixe_couvert = prix_actuel * 1.02 # Légère prime de couverture
    prix_avec_couverture = (prix_sans_couverture * (1 - taux_couverture)) + (prix_fixe_couvert * taux_couverture)
    
    # Graphique Matplotlib aux couleurs Air Algérie
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mois, prix_sans_couverture, label="Sans Couverture (Subi)", color="#333333", linestyle="--", marker="o")
    ax.plot(mois, prix_avec_couverture, label="Avec Couverture (Lissé)", color="#D2143A", linewidth=2.5, marker="s")
    ax.set_title("Impact du Hedging sur le Budget Carburant (USD/Baril)", color="#D2143A")
    ax.set_xlabel("Mois de l'exercice")
    ax.set_ylabel("Prix du Baril (USD)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    
    st.pyplot(fig)
    
    # Métriques financières
    col1, col2, col3 = st.columns(3)
    col1.metric("Risque Maximum Évité", f"{max(prix_sans_couverture) - max(prix_avec_couverture):.2f} USD / baril")
    col2.metric("Économie Potentielle Estimée", f"{(prix_sans_couverture.mean() - prix_avec_couverture.mean()) * 50000:,.2f} DA", delta="Optimisé")
    col3.metric("Stabilité de la Trésorerie", f"+ {taux_couverture*100:.0f}% de visibilité")

# --- ONGLET 2 : ACTUARIAT (MODÉLISATION DES RETARDS) ---
with tab2:
    st.header("Modélisation Actuarielle de la Charge Sinistre (Retards)")
    st.write("Utilisation d'une Loi de Poisson pour la fréquence et d'une distribution de Sévérité pour estimer les provisions financières nécessaires.")
    
    # Simulation actuarielle de la charge totale (Modèle Collectif)
    # Nombre de sinistres (Retards) sur 1000 simulations
    sim_frequence = np.random.poisson(lambda_poisson, 1000)
    # Coût total simulé
    sim_charge_totale = []
    for freq in sim_frequence:
        # Sévérité suit une loi log-normale (asymétrique, typique en actuariat)
        severites = np.random.lognormal(mean=np.log(cout_moyen) - 0.2, sigma=0.5, size=freq)
        sim_charge_totale.append(severites.sum())
        
    sim_charge_totale = np.array(sim_charge_totale)
    
    # Calcul des indicateurs actuariels clés
    charge_moyenne = sim_charge_totale.mean()
    VaR_95 = np.percentile(sim_charge_totale, 95) # Value at Risk 95%
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribution de la Charge Financière Annuelle")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(sim_charge_totale / 1e6, bins=30, color="#D2143A", alpha=0.8, edgecolor="white")
        ax2.axvline(VaR_95 / 1e6, color="#333333", linestyle="--", label=f"VaR 95% ({VaR_95/1e6:.1f}M DA)")
        ax2.set_xlabel("Charge Totale Mensuelle (En Millions de DA)")
        ax2.set_ylabel("Fréquence (Simulations)")
        ax2.legend()
        st.pyplot(fig2)
        
    with col2:
        st.subheader("🛡️ Évaluation des Besoins en Provisions")
        st.write("En se basant sur les principes actuariels (similaires à Solvabilité II) :")
        
        st.metric("Provision Pure (Charge attendue)", f"{charge_moyenne:,.2f} DA")
        st.metric("Capital Économique Requis (VaR 95%)", f"{VaR_95:,.2f} DA")
        
        st.info("💡 **Note Méthodologique (Double Machine Learning) :** En intégrant un modèle DML, nous pouvons isoler l'effet causal strict des pannes techniques d'Air Algérie sur cette charge totale, en éliminant les biais de confusion saisonniers (Saison Estivale, Hadj, Omra).")