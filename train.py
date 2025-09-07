import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

# 1. Connexion à MongoDB
client = MongoClient("mongodb://localhost:27017/")
db_weather = client["weather_data"]
db_satellite = client["satellite_data"]
db_agro = client["agro_monitoring"]

# 2. Extraction des données avec gestion des doublons
def extract_mongo_data():
    """Récupère les données de toutes les collections et les fusionne"""
    # Données météo - Agrégation par date et région
    weather = list(db_weather.regional_weather.find({}, {
        "region": 1,
        "datetime": 1,
        "temperature": 1,
        "humidity": 1,
        "_id": 0
    }))
    df_weather = pd.DataFrame(weather)
    df_weather['date'] = pd.to_datetime(df_weather['datetime']).dt.normalize()
    
    # Agrégation des données météo (moyenne par jour)
    df_weather = df_weather.groupby(['region', 'date']).agg({
        'temperature': 'mean',
        'humidity': 'mean'
    }).reset_index()

    # Données satellites - Vérification des doublons
    satellite = list(db_satellite.regional_satellite.find({}, {
        "region": 1,
        "date": 1,
        "mean_ndvi": 1,
        "mean_ndwi": 1,
        "_id": 0
    }))
    df_satellite = pd.DataFrame(satellite)
    df_satellite['date'] = pd.to_datetime(df_satellite['date']).dt.normalize()
    
    # Suppression des doublons satellites (garder le dernier)
    df_satellite = df_satellite.drop_duplicates(
        subset=['region', 'date'],
        keep='last'
    )

    # Données agro quotidiennes - Extraction nested fields
    agro_daily = list(db_agro.daily_region_stats.find({}, {
        "region": 1,
        "date": 1,
        "data.soil.moisture": 1,
        "data.temperature.surface": 1,
        "_id": 0
    }))
    df_agro_daily = pd.DataFrame(agro_daily)
    df_agro_daily['date'] = pd.to_datetime(df_agro_daily['date']).dt.normalize()
    df_agro_daily.rename(columns={
        "data.soil.moisture": "soil_moisture",
        "data.temperature.surface": "soil_temp"
    }, inplace=True)
    
    # Suppression des doublons agro (garder le dernier)
    df_agro_daily = df_agro_daily.drop_duplicates(
        subset=['region', 'date'],
        keep='last'
    )

    # Fusion progressive
    try:
        # Fusion météo + satellite
        df = pd.merge(
            df_weather,
            df_satellite,
            on=['region', 'date'],
            how='outer'  # Garder toutes les données même sans correspondance
        )
        
        # Fusion avec données agro
        df = pd.merge(
            df,
            df_agro_daily,
            on=['region', 'date'],
            how='outer'
        )
        
        return df.dropna()
    
    except Exception as e:
        print(f"Erreur de fusion: {str(e)}")
        print("\nDebug - Taille des DataFrames:")
        print(f"Météo: {len(df_weather)} | Satellite: {len(df_satellite)} | Agro: {len(df_agro_daily)}")
        print("\nExemples de doublons:")
        print("Satellite:", df_satellite[df_satellite.duplicated(['region', 'date'], keep=False)])
        print("Agro:", df_agro_daily[df_agro_daily.duplicated(['region', 'date'], keep=False)])
        raise

# ... (le reste des fonctions prepare_features, prepare_training_data, train_model reste identique)

# 3. Pipeline complet avec vérification des données
def main():
    print("Début du traitement des données...")
    
    try:
        # Étape 1: Extraction et fusion
        df = extract_mongo_data()
        print(f"\nDonnées fusionnées ({len(df)} lignes)")
        print("Période couverte:", df['date'].min(), "à", df['date'].max())
        print("Régions:", df['region'].unique())
        
        # Étape 2: Feature engineering
        df_processed = prepare_features(df)
        print("\nFeatures disponibles:", list(df_processed.columns))
        
        # Étape 3: Préparation entraînement
        X, y, scaler = prepare_training_data(df_processed)
        print(f"\nDistribution des classes:\n{y.value_counts()}")
        
        # Étape 4: Entraînement
        print("\nEntraînement du modèle...")
        model = train_model(X, y)
        
        # Étape 5: Sauvegarde
        joblib.dump(model, 'crisis_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("\nModèle et scaler sauvegardés")
        
        # Visualisation
        plot_feature_importance(model)
        
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        print("Conseils:")
        print("- Vérifiez les doublons dans vos collections MongoDB")
        print("- Inspectez les données avec df.head() à chaque étape")
        print("- Consultez les logs ci-dessus pour diagnostiquer")

def plot_feature_importance(model):
    """Visualisation des caractéristiques importantes"""
    feat_importances = pd.Series(model.feature_importances_, index=[
        'Température', 'Humidité', 'NDVI', 'NDWI',
        'Humidité sol', 'Temp sol', 'Jour année',
        'Temp 7j moy', 'NDVI 30j moy', 'Stress hydrique'
    ])
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Importance des caractéristiques')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

if __name__ == "__main__":
    main()