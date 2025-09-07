import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="🌱 AgroPredict - Diagnostic Végétal",
    page_icon="🌾",
    layout="wide"
)

# Données des régions
REGIONS = [
    {"name": "Dakar", "lat": 14.7167, "lon": -17.4677},
    {"name": "Thiès", "lat": 14.8136, "lon": -16.9622},
    {"name": "Saint-Louis", "lat": 16.0179, "lon": -16.4896},
    {"name": "Ziguinchor", "lat": 12.5642, "lon": -16.2719},
    {"name": "Kaolack", "lat": 14.1828, "lon": -16.2534},
    {"name": "Tambacounda", "lat": 13.7708, "lon": -13.6673},
    {"name": "Matam", "lat": 15.6553, "lon": -13.2556},
    {"name": "Kolda", "lat": 12.8927, "lon": -14.9389},
    {"name": "Fatick", "lat": 14.3431, "lon": -16.4183},
    {"name": "Louga", "lat": 15.6100, "lon": -16.2310},
    {"name": "Kaffrine", "lat": 14.1052, "lon": -15.5434},
    {"name": "Diourbel", "lat": 14.6605, "lon": -16.2366},
    {"name": "Sédhiou", "lat": 12.7089, "lon": -15.5560},
    {"name": "Kédougou", "lat": 12.5611, "lon": -12.1740},
    {"name": "Podor", "lat": 16.6566, "lon": -14.9612}
]

# Chargement du modèle
@st.cache_resource
def load_model():
    return joblib.load('modele/agriculture_model_pipeline.pkl')

model = load_model()

# Interface Utilisateur
st.title("🌿 AgroPredict - Analyse NDVI en Temps Réel")
st.markdown("""
    *Prédisez la santé des cultures grâce à l'IA*  
    Ce modèle évalue l'indice NDVI (Normalized Difference Vegetation Index) à partir des conditions environnementales.
    Le NDVI est un indicateur numérique qui quantifie la vigueur de la végétation en utilisant les valeurs extraites des images satellites, sans nécessiter la manipulation directe de ces images.
""")

with st.expander("ℹ️ Comment utiliser"):
    st.write("""
    1. Sélectionnez votre région
    2. Ajustez les paramètres météo et sol
    3. Cliquez sur 'Prédire'
    4. Consultez l'analyse détaillée
    """)

# Sidebar pour les paramètres
with st.sidebar:
    st.header("📊 Paramètres d'Entrée")
    
    # Sélection de la région avec autocomplétion
    selected_region = st.selectbox(
        "Région",
        options=[r["name"] for r in REGIONS],
        index=0
    )
    
    # Récupération automatique des coordonnées
    region_data = next(r for r in REGIONS if r["name"] == selected_region)
    latitude = region_data["lat"]
    longitude = region_data["lon"]
    
    # Affichage des coordonnées (lecture seule)
    st.markdown(f"""
    **Coordonnées GPS:**
    - Latitude: `{latitude}`
    - Longitude: `{longitude}`
    """)
    
    st.subheader("Conditions Météo")
    temperature = st.slider("Température (°C)", -10.0, 50.0, 25.0, 0.1)
    humidity = st.slider("Humidité Relative (%)", 0, 100, 60)
    wind_speed = st.slider("Vitesse du Vent (km/h)", 0.0, 100.0, 10.0, 0.1)
    
    st.subheader("État du Sol")
    soil_moisture = st.slider("Humidité du Sol (0-1)", 0.0, 1.0, 0.2, 0.01)
    
    st.subheader("Date d'Observation")
    obs_date = st.date_input("Date", datetime.now())
    
    predict_btn = st.button("🌱 Prédire l'État des Cultures", type="primary")

# Calcul des features dérivées
day_of_year = obs_date.timetuple().tm_yday
day_sin = np.sin(2 * np.pi * day_of_year/365)
day_cos = np.cos(2 * np.pi * day_of_year/365)
season = (obs_date.month % 12 + 3) // 3  # 1:Hiver, 2:Printemps, etc.

# Préparation des données
input_data = pd.DataFrame({
    'temperature': [temperature],
    'humidity': [humidity],
    'wind_speed': [wind_speed],
    'soil_moisture': [soil_moisture],
    'longitude': [longitude],
    'latitude': [latitude],
    'day_sin': [day_sin],
    'day_cos': [day_cos],
    'season': [season],
    'temp_range': [10]  # Valeur par défaut
})

# Prédiction et affichage
if predict_btn:
    with st.spinner('Analyse en cours...'):
        ndvi_pred = model.predict(input_data)[0]
        
        # Affichage des résultats
        st.success("Analyse terminée !")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Résultat de Prédiction")
            st.metric(label="**NDVI Prédit**", value=f"{ndvi_pred:.3f}")
            
            # Jauge visuelle
            st.progress(float(np.clip(ndvi_pred, 0, 1)))
            
            # Interprétation
            if ndvi_pred < 0.1:
                st.error("🚨 Alerte : Stress Végétal Sévère")
                st.markdown("""
                **Recommandations :**
                - Irrigation immédiate requise
                - Vérifier les parasites
                - Considérer des cultures plus résistantes
                """)
            elif 0.1 <= ndvi_pred < 0.3:
                st.warning("⚠️ Attention : Croissance Ralentie")
                st.markdown("""
                **Recommandations :**
                - Surveiller l'humidité du sol
                - Fertilisation modérée recommandée
                """)
            else:
                st.success("✅ Statut Optimal")
                st.markdown("""
                **Recommandations :**
                - Poursuivre les pratiques actuelles
                - Surveillance régulière conseillée
                """)
        
        with col2:
            st.subheader("Détails Techniques")
            st.markdown(f"""
            **Paramètres utilisés :**
            - Région: {selected_region}
            - Coordonnées: {latitude:.4f}°N, {longitude:.4f}°E
            - Date: {obs_date.strftime('%d/%m/%Y')} (Saison {season})
            - Température: {temperature}°C
            - Humidité sol: {soil_moisture*100:.1f}%
            """)
            
            # Carte miniature (optionnelle)
            st.map(pd.DataFrame({
                'lat': [latitude],
                'lon': [longitude]
            }), zoom=6)