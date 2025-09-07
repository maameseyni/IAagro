import ee
import time
from pymongo import MongoClient
from datetime import datetime, timedelta
import warnings
import traceback
from tqdm import tqdm

# Configuration initiale
warnings.filterwarnings("ignore")
ee.Initialize(project='ee-imamkebedev')

# Connexion MongoDB
client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
db = client["agro_monitoring"]
collection = db["daily_region_stats"]

# Connexion MongoDB
#client = MongoClient("mongodb+srv://imam:imam@cluster0.ny4c0ui.mongodb.net/")
#db = client["agro_monitoring"]
#collection = db["daily_region_stats"]

# Paramètres
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=180)
BUFFER_RADIUS = 20000  # 20 km
MAX_CLOUD_COVER = 60  # %

# Collections optimisées
COLLECTIONS = {
    'sentinel': 'COPERNICUS/S2_SR_HARMONIZED',
    'smap': 'NASA_USDA/HSL/SMAP10KM_soil_moisture',
    'modis': 'MODIS/061/MOD11A1',
    'era5': 'ECMWF/ERA5_LAND/HOURLY'
}

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



def get_daily_data(region, date):
    """Récupère toutes les données pour une région à une date donnée"""
    point = ee.Geometry.Point(region["lon"], region["lat"]).buffer(BUFFER_RADIUS)
    date_str = date.strftime('%Y-%m-%d')
    next_day = date + timedelta(days=1)
    
    doc = {
        "_id": f"{region['name']}_{date_str}",
        "region": region["name"],
        "date": date_str,
        "coordinates": [region["lon"], region["lat"]],
        "data": {}
    }

    # 1. Données Sentinel-2 (Végétation)
    try:
        s2 = ee.ImageCollection(COLLECTIONS['sentinel']) \
              .filterBounds(point) \
              .filterDate(date_str, next_day.strftime('%Y-%m-%d')) \
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER)) \
              .sort('CLOUDY_PIXEL_PERCENTAGE') \
              .first()
        
        if s2:
            ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
            evi = s2.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                    'NIR': s2.select('B8'),
                    'RED': s2.select('B4'),
                    'BLUE': s2.select('B2')
                }).rename('EVI')
            
            stats = ndvi.addBands(evi).reduceRegion(
                ee.Reducer.mean(), point, 100).getInfo()
            
            doc["data"]["vegetation"] = {
                "ndvi": stats.get('NDVI'),
                "evi": stats.get('EVI'),
                "cloud_cover": s2.get('CLOUDY_PIXEL_PERCENTAGE').getInfo(),
                "source": "Sentinel-2"
            }
    except Exception as e:
        print(f"Erreur Sentinel-2 pour {region['name']} {date_str}: {str(e)}")

    # 2. Données SMAP (Humidité du sol)
    try:
        smap = ee.ImageCollection(COLLECTIONS['smap']) \
                .filterBounds(point) \
                .filterDate(date_str, next_day.strftime('%Y-%m-%d')) \
                .first()
        
        if smap:
            moisture = smap.select('ssm').reduceRegion(
                ee.Reducer.mean(), point, 10000).getInfo()
            
            if moisture['ssm'] is not None:
                doc["data"]["soil"] = {
                    "moisture": moisture['ssm'],
                    "source": "SMAP"
                }
    except:
        try:
            # Fallback sur ERA5
            era5 = ee.ImageCollection(COLLECTIONS['era5']) \
                    .filterBounds(point) \
                    .filterDate(date_str, next_day.strftime('%Y-%m-%d')) \
                    .first()
            
            if era5:
                moisture = era5.select('volumetric_soil_water_layer_1') \
                              .reduceRegion(ee.Reducer.mean(), point, 10000) \
                              .getInfo()
                
                if moisture['volumetric_soil_water_layer_1'] is not None:
                    doc["data"]["soil"] = {
                        "moisture": moisture['volumetric_soil_water_layer_1'],
                        "source": "ERA5"
                    }
        except Exception as e:
            print(f"Erreur humidité pour {region['name']} {date_str}: {str(e)}")

    # 3. Données MODIS (Température)
    try:
        modis = ee.ImageCollection(COLLECTIONS['modis']) \
                 .filterBounds(point) \
                 .filterDate(date_str, next_day.strftime('%Y-%m-%d')) \
                 .first()
        
        if modis:
            lst = modis.select('LST_Day_1km').reduceRegion(
                ee.Reducer.mean(), point, 1000).getInfo()
            
            if lst['LST_Day_1km'] is not None:
                doc["data"]["temperature"] = {
                    "surface": lst['LST_Day_1km'] * 0.02 - 273.15,
                    "source": "MODIS"
                }
    except Exception as e:
        print(f"Erreur MODIS pour {region['name']} {date_str}: {str(e)}")

    # Ne sauvegarder que si on a au moins une donnée
    if doc["data"]:
        try:
            collection.update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
            return True
        except Exception as e:
            print(f"Erreur MongoDB pour {region['name']} {date_str}: {str(e)}")
    
    return False

def process_region(region):
    """Traite tous les jours pour une région"""
    current_date = START_DATE
    success_count = 0
    
    with tqdm(total=180, desc=f"Région {region['name']}", unit="jour") as pbar:
        while current_date <= END_DATE:
            try:
                if get_daily_data(region, current_date):
                    success_count += 1
            except Exception as e:
                print(f"Erreur majeure pour {region['name']} {current_date}:")
                traceback.print_exc()
                time.sleep(60)  # Pause longue en cas d'erreur
            
            current_date += timedelta(days=1)
            pbar.update(1)
            time.sleep(1)  # Pause entre les requêtes
    
    return success_count

def main():
    print(f"\n🚀 Début du traitement quotidien ({START_DATE} à {END_DATE})")
    
    total_days = 0
    for region in REGIONS:
        print(f"\n🌍 Traitement de {region['name']}...")
        success = process_region(region)
        total_days += success
        print(f"✅ {success} jours enregistrés pour {region['name']}")
        time.sleep(5)  # Pause entre les régions
    
    print(f"\n✨ Terminé! {total_days} jours de données enregistrés au total")

if __name__ == "__main__":
    try:
        main()
    finally:
        client.close()