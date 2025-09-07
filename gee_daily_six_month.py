import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
import time
from tqdm import tqdm
import warnings

# Désactivation des avertissements
warnings.filterwarnings("ignore")

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
    db = client["agro_daily"]
    collection = db["region_gee_daily_data"]
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
START_DATE = END_DATE - timedelta(days=180)  # 6 mois
BUFFER_RADIUS = 10000  # 10km
MAX_CLOUD_COVER = 70  # %

def get_daily_data(region, date):
    """Récupère les données quotidiennes pour une région"""
    point = ee.Geometry.Point(region["lon"], region["lat"]).buffer(BUFFER_RADIUS)
    date_str = date.strftime('%Y-%m-%d')
    doc_id = f"{region['name']}_{date_str}"
    
    # Vérifier si la date existe déjà
    if collection.count_documents({"_id": doc_id}, limit=1):
        return None

    doc = {
        "_id": doc_id,
        "region": region["name"],
        "date": date_str,
        "coordinates": [region["lon"], region["lat"]],
        "data": {},
        "errors": {}
    }

    # 1. Données Sentinel-2 (Végétation)
    try:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
              .filterBounds(point) \
              .filterDate(date_str, (date + timedelta(days=1)).strftime('%Y-%m-%d')) \
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER)) \
              .sort('CLOUDY_PIXEL_PERCENTAGE') \
              .first()
        
        if s2:
            ndvi = s2.normalizedDifference(['B8', 'B4'])
            evi = s2.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {'NIR': s2.select('B8'), 'RED': s2.select('B4'), 'BLUE': s2.select('B2')}
            )
            
            stats = ndvi.addBands(evi).reduceRegion(
                ee.Reducer.mean(),
                point,
                100
            ).getInfo()
            
            doc["data"]["vegetation"] = {
                "ndvi": stats.get('NDVI'),
                "evi": stats.get('EVI'),
                "cloud_cover": s2.get('CLOUDY_PIXEL_PERCENTAGE').getInfo(),
                "source": "Sentinel-2"
            }
        else:
            doc["errors"]["vegetation"] = "Aucune image disponible ou couverture nuageuse trop élevée"
    except Exception as e:
        doc["errors"]["vegetation"] = str(e)

    # 2. Données ERA5 (Météo)
    try:
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
                .filterBounds(point) \
                .filterDate(date_str, (date + timedelta(days=1)).strftime('%Y-%m-%d')) \
                .select(['temperature_2m', 'volumetric_soil_water_layer_1', 'total_precipitation']) \
                .mean()
        
        era5_stats = era5.reduceRegion(
            ee.Reducer.mean(),
            point,
            10000
        ).getInfo()
        
        doc["data"]["weather"] = {
            "temperature": era5_stats.get('temperature_2m') - 273.15 if era5_stats.get('temperature_2m') else None,  # K → °C
            "soil_moisture": era5_stats.get('volumetric_soil_water_layer_1'),
            "precipitation": era5_stats.get('total_precipitation') * 1000 if era5_stats.get('total_precipitation') else None,  # m → mm
            "source": "ERA5"
        }
    except Exception as e:
        doc["errors"]["weather"] = str(e)

    # 3. Données MODIS (Température de surface)
    try:
        modis = ee.ImageCollection("MODIS/061/MOD11A1") \
                 .filterBounds(point) \
                 .filterDate(date_str, (date + timedelta(days=1)).strftime('%Y-%m-%d')) \
                 .first()
        
        if modis:
            lst = modis.select('LST_Day_1km').reduceRegion(
                ee.Reducer.mean(),
                point,
                1000
            ).getInfo()
            
            doc["data"]["surface_temp"] = {
                "value": lst.get('LST_Day_1km') * 0.02 - 273.15 if lst.get('LST_Day_1km') else None,  # Conversion en °C
                "source": "MODIS"
            }
        else:
            doc["errors"]["surface_temp"] = "Aucune donnée disponible"
    except Exception as e:
        doc["errors"]["surface_temp"] = str(e)

    # Ne garder que les documents avec des données valides
    if doc["data"]:
        try:
            collection.update_one({"_id": doc_id}, {"$set": doc}, upsert=True)
            return doc
        except Exception as e:
            print(f"❌ Erreur MongoDB: {str(e)}")
    
    return None

def process_region(region):
    """Traite tous les jours pour une région"""
    current_date = START_DATE
    success_count = 0
    
    with tqdm(total=(END_DATE - START_DATE).days + 1, desc=f"Région {region['name']}", unit="jour") as pbar:
        while current_date <= END_DATE:
            try:
                result = get_daily_data(region, current_date)
                if result:
                    success_count += 1
            except Exception as e:
                print(f"❌ Erreur majeure {region['name']} {current_date}: {str(e)}")
                time.sleep(60)  # Longue pause après erreur
            
            current_date += timedelta(days=1)
            pbar.update(1)
            time.sleep(1)  # Pause courte entre les jours
    
    return success_count

def export_to_dataframe():
    """Exporte les données vers un DataFrame pandas"""
    cursor = collection.find({})
    df = pd.DataFrame(list(cursor))
    
    if not df.empty:
        # Extraction des features
        weather_df = pd.json_normalize(df['data'].apply(lambda x: x.get('weather', {})))
        vegetation_df = pd.json_normalize(df['data'].apply(lambda x: x.get('vegetation', {})))
        surface_temp_df = pd.json_normalize(df['data'].apply(lambda x: x.get('surface_temp', {})))
        
        # Fusion
        final_df = pd.concat([
            df[['_id', 'region', 'date', 'coordinates']],
            weather_df,
            vegetation_df,
            surface_temp_df
        ], axis=1)
        
        # Nettoyage
        final_df['date'] = pd.to_datetime(final_df['date'])
        final_df = final_df.dropna(subset=['ndvi', 'temperature'])  # Garder seulement les lignes avec les features essentielles
        
        # Feature engineering
        final_df['day_of_year'] = final_df['date'].dt.dayofyear
        final_df['day_sin'] = np.sin(2 * np.pi * final_df['day_of_year']/365)
        final_df['day_cos'] = np.cos(2 * np.pi * final_df['day_of_year']/365)
        
        # Lag features
        for lag in [1, 7, 30]:# 1 jour, 1 semaine, 1 mois
            final_df[f'ndvi_lag_{lag}'] = final_df.groupby('region')['ndvi'].shift(lag)
        
        return final_df.dropna()
    
    return pd.DataFrame()

def main():
    print(f"\n🌾 Début du traitement quotidien ({START_DATE.date()} au {END_DATE.date()})")
    
    # Traitement par région
    total_days = 0
    for region in REGIONS:
        print(f"\n🌍 Traitement de {region['name']}...")
        success = process_region(region)
        total_days += success
        print(f"✅ {success} jours traités pour {region['name']}")
        time.sleep(5)  # Pause entre les régions
    
    # Export pour le ML
    print("\n📊 Export des données pour le machine learning...")
    ml_df = export_to_dataframe()
    
    if not ml_df.empty:
        ml_df.to_parquet("daily_agro_data.parquet")
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