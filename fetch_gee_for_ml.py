import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient
import time
from tqdm import tqdm

# Initialisation Earth Engine
try:
    ee.Initialize(project='ee-imamkebedev')
    print("✅ Earth Engine initialisé")
except Exception as e:
    print(f"❌ Erreur Earth Engine: {e}")
    exit()

# Connexion MongoDB
try:
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
    db = client["agro_analytics"]
    collection = db["ml_ready_data"]
    print("✅ Connecté à MongoDB")
except Exception as e:
    print(f"❌ Erreur MongoDB: {e}")
    exit()

# Configuration
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

END_DATE = datetime.now()
START_DATE = END_DATE - relativedelta(months=24)  # 2 ans de données
BUFFER_RADIUS = 15000  # 15km

def get_monthly_features(region, year, month):
    """Récupère les caractéristiques mensuelles pour le ML"""
    point = ee.Geometry.Point(region["lon"], region["lat"]).buffer(BUFFER_RADIUS)
    start_date = datetime(year, month, 1)
    end_date = start_date + relativedelta(months=1)
    date_str = start_date.strftime('%Y-%m-%d')
    
    # 1. Données de végétation (Sentinel-2 + MODIS fallback)
    try:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
              .filterBounds(point) \
              .filterDate(start_date, end_date) \
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
              .mean()
        
        if 'B8' in s2.bandNames().getInfo():
            ndvi = s2.normalizedDifference(['B8', 'B4'])
            ndvi_val = ndvi.reduceRegion(ee.Reducer.mean(), point, 1000).getInfo()
            ndvi = list(ndvi_val.values())[0] if ndvi_val else None
        else:
            modis = ee.ImageCollection("MODIS/006/MOD13A2") \
                     .filterBounds(point) \
                     .filterDate(start_date, end_date) \
                     .first()
            ndvi = modis.select('NDVI').reduceRegion(ee.Reducer.mean(), point, 1000).get('NDVI').getInfo() if modis else None
    except:
        ndvi = None

    # 2. Données ERA5 (météo)
    try:
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY") \
                .filterBounds(point) \
                .filterDate(start_date, end_date) \
                .first()
        
        if era5:
            temp = era5.select('temperature_2m').reduceRegion(ee.Reducer.mean(), point, 10000).get('temperature_2m').getInfo()
            moisture = era5.select('volumetric_soil_water_layer_1').reduceRegion(ee.Reducer.mean(), point, 10000).get('volumetric_soil_water_layer_1').getInfo()
            precip = era5.select('total_precipitation').reduceRegion(ee.Reducer.mean(), point, 10000).get('total_precipitation').getInfo()
        else:
            temp = moisture = precip = None
    except:
        temp = moisture = precip = None

    # Construction du document
    doc = {
        "_id": f"{region['name']}_{year}_{month:02d}",
        "region": region["name"],
        "year": year,
        "month": month,
        "features": {
            "ndvi": ndvi,
            "temperature": temp - 273.15 if temp else None,  # Conversion K → °C
            "soil_moisture": moisture,
            "precipitation": precip * 1000 if precip else None  # Conversion m → mm
        },
        "processed_at": datetime.now().isoformat()
    }
    
    # Nettoyage des valeurs nulles
    doc["features"] = {k: v for k, v in doc["features"].items() if v is not None}
    
    return doc if doc["features"] else None

def save_to_mongodb(doc):
    """Sauvegarde les données dans MongoDB"""
    try:
        collection.update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
        return True
    except Exception as e:
        print(f"❌ Erreur MongoDB: {str(e)}")
        return False

def export_to_ml_format():
    """Exporte les données vers un format prêt pour le ML"""
    cursor = collection.find({})
    df = pd.DataFrame(list(cursor))
    
    if not df.empty:
        # Transformation des features
        features_df = pd.json_normalize(df['features'])
        final_df = pd.concat([df[['region', 'year', 'month']], features_df], axis=1)
        
        # Feature engineering
        final_df['date'] = pd.to_datetime(final_df[['year', 'month']].assign(day=1))
        final_df['month_sin'] = np.sin(2 * np.pi * final_df['month']/12)
        final_df['month_cos'] = np.cos(2 * np.pi * final_df['month']/12)
        
        # Lag features
        for lag in [1, 2, 3, 12]:
            final_df[f'ndvi_lag_{lag}'] = final_df.groupby('region')['ndvi'].shift(lag)
        
        return final_df.dropna()
    return pd.DataFrame()

def main():
    print(f"\n🌾 Début du traitement ({START_DATE.year}-{START_DATE.month} à {END_DATE.year}-{END_DATE.month})")
    
    # 1. Récupération des données
    current_date = START_DATE.replace(day=1)
    while current_date <= END_DATE:
        for region in REGIONS:
            try:
                doc = get_monthly_features(
                    region,
                    current_date.year,
                    current_date.month
                )
                if doc:
                    save_to_mongodb(doc)
                    print(f"✅ {region['name']} {current_date.year}-{current_date.month:02d}")
                else:
                    print(f"⚠️ Données manquantes pour {region['name']} {current_date.year}-{current_date.month:02d}")
                time.sleep(1)  # Respect des quotas Earth Engine
            except Exception as e:
                print(f"❌ Erreur {region['name']} {current_date.year}-{current_date.month:02d}: {str(e)}")
                time.sleep(5)
        
        current_date += relativedelta(months=1)
    
    # 2. Export pour le ML
    print("\n📊 Export des données pour le machine learning...")
    ml_df = export_to_ml_format()
    
    if not ml_df.empty:
        ml_df.to_parquet("senegal_agro_data.parquet")
        print(f"\n✅ Export réussi! {len(ml_df)} enregistrements")
        print("🔍 Aperçu des données:")
        print(ml_df.head())
    else:
        print("❌ Aucune donnée à exporter")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Traitement interrompu")
    finally:
        client.close()