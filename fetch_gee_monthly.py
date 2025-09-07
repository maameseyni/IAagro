import ee
import time
from datetime import datetime, timedelta
from pymongo import MongoClient
from tqdm import tqdm
import warnings
import traceback
from dateutil.relativedelta import relativedelta

# Initialisation Earth Engine
ee.Initialize(project='ee-imamkebedev')

# Configuration MongoDB
client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
db = client["agro_monitoring"]
collection = db["monthly_complete_stats"]

# Connexion MongoDB
#client = MongoClient("mongodb+srv://imam:imam@cluster0.ny4c0ui.mongodb.net/")
#db = client["agro_monitoring"]
#collection = db["daily_region_stats"]

# Paramètres
END_DATE = datetime.now()
START_DATE = END_DATE - relativedelta(months=6)
BUFFER_RADIUS = 15000  # 15km de rayon
MAX_CLOUD_COVER = 80  # % plus permissif

# Collections fiables
COLLECTIONS = {
    'vegetation': {
        'primary': 'COPERNICUS/S2_SR_HARMONIZED',
        'fallback': 'MODIS/006/MOD13A2'  # NDVI MODIS si Sentinel échoue
    },
    'soil': {
        'primary': 'ECMWF/ERA5_LAND/MONTHLY',  # Données mensuelles directes
        'fallback': 'ECMWF/ERA5_LAND/HOURLY'   # Calcul manuel si nécessaire
    },
    'temperature': {
        'primary': 'ECMWF/ERA5_LAND/MONTHLY',
        'fallback': 'MODIS/061/MOD11A2'  # LST 8 jours
    }
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

def get_monthly_image(collection_name, region_geom, year, month):
    """Récupère l'image mensuelle avec re-tentatives"""
    start_date = datetime(year, month, 1)
    end_date = (start_date + relativedelta(months=1)).strftime('%Y-%m-%d')
    start_date = start_date.strftime('%Y-%m-%d')
    
    for attempt in range(3):
        try:
            coll = ee.ImageCollection(COLLECTIONS[collection_name]['primary']) \
                  .filterBounds(region_geom) \
                  .filterDate(start_date, end_date)
            
            if collection_name == 'vegetation':
                coll = coll.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER))
            
            img = coll.mean()  # Moyenne mensuelle
            
            if img.bandNames().size().getInfo() == 0:
                raise ValueError("No data available")
                
            return img
        
        except Exception as e:
            if attempt == 2:  # Dernière tentative
                print(f"⚠️ Échec {collection_name} ({year}-{month}), tentative fallback...")
                try:
                    fallback_coll = ee.ImageCollection(COLLECTIONS[collection_name]['fallback']) \
                                   .filterBounds(region_geom) \
                                   .filterDate(start_date, end_date)
                    
                    if collection_name == 'vegetation':
                        fallback_coll = fallback_coll.filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD_COVER))
                    
                    return fallback_coll.mean()
                except:
                    return None
            time.sleep(5)

def process_region_month(region, year, month):
    """Traite une région pour un mois donné"""
    region_geom = ee.Geometry.Point(region["lon"], region["lat"]).buffer(BUFFER_RADIUS)
    month_str = f"{year}-{month:02d}"
    doc_id = f"{region['name']}_{month_str}"
    
    # Vérifier si le mois existe déjà
    if collection.count_documents({"_id": doc_id}, limit=1):
        return False

    doc = {
        "_id": doc_id,
        "region": region["name"],
        "year": year,
        "month": month,
        "coordinates": [region["lon"], region["lat"]],
        "data": {},
        "metadata": {
            "processed_at": datetime.now().isoformat(),
            "status": "complete"
        }
    }

    # 1. Végétation (NDVI)
    veg_img = get_monthly_image('vegetation', region_geom, year, month)
    if veg_img:
        try:
            if 'NDVI' in veg_img.bandNames().getInfo():  # MODIS
                ndvi = veg_img.select('NDVI')
            else:  # Sentinel-2
                ndvi = veg_img.normalizedDifference(['B8', 'B4'])
            
            ndvi_val = ndvi.reduceRegion(
                ee.Reducer.mean(), 
                region_geom, 
                1000  # Résolution plus large pour les moyennes
            ).getInfo()
            
            doc["data"]["vegetation"] = {
                "ndvi": list(ndvi_val.values())[0],
                "source": veg_img.get('system:index').getInfo() or "Sentinel-2/MODIS"
            }
        except Exception as e:
            doc["metadata"]["status"] = "partial"
            doc["metadata"]["vegetation_error"] = str(e)

    # 2. Humidité du sol
    soil_img = get_monthly_image('soil', region_geom, year, month)
    if soil_img:
        try:
            band = 'volumetric_soil_water_layer_1'
            moisture = soil_img.select(band).reduceRegion(
                ee.Reducer.mean(), 
                region_geom, 
                10000
            ).get(band).getInfo()
            
            doc["data"]["soil"] = {
                "moisture": moisture,
                "source": "ERA5"
            }
        except Exception as e:
            doc["metadata"]["status"] = "partial"
            doc["metadata"]["soil_error"] = str(e)

    # 3. Température
    temp_img = get_monthly_image('temperature', region_geom, year, month)
    if temp_img:
        try:
            if 'LST_Day_1km' in temp_img.bandNames().getInfo():  # MODIS
                lst = temp_img.select('LST_Day_1km')
                temp_val = lst.reduceRegion(
                    ee.Reducer.mean(),
                    region_geom,
                    1000
                ).get('LST_Day_1km').getInfo()
                temp_c = temp_val * 0.02 - 273.15 if temp_val else None
            else:  # ERA5
                temp_val = temp_img.select('temperature_2m').reduceRegion(
                    ee.Reducer.mean(),
                    region_geom,
                    10000
                ).get('temperature_2m').getInfo()
                temp_c = temp_val - 273.15 if temp_val else None
            
            if temp_c:
                doc["data"]["temperature"] = {
                    "mean": temp_c,
                    "source": "ERA5" if 'temperature_2m' in temp_img.bandNames().getInfo() else "MODIS"
                }
        except Exception as e:
            doc["metadata"]["status"] = "partial"
            doc["metadata"]["temp_error"] = str(e)

    # Vérification complétude
    if not all(k in doc["data"] for k in ['vegetation', 'soil', 'temperature']):
        doc["metadata"]["status"] = "partial"

    # Sauvegarde
    try:
        collection.update_one({"_id": doc_id}, {"$set": doc}, upsert=True)
        return True
    except Exception as e:
        print(f"❌ Erreur MongoDB: {str(e)}")
        return False

def main():
    print(f"\n🚀 Début du traitement mensuel ({START_DATE.year}-{START_DATE.month} à {END_DATE.year}-{END_DATE.month})")
    
    # Générer tous les mois à traiter
    current = START_DATE.replace(day=1)
    months_to_process = []
    while current <= END_DATE:
        months_to_process.append((current.year, current.month))
        current += relativedelta(months=1)
    
    # Traitement par région et mois
    for region in REGIONS:
        print(f"\n🌍 Traitement de {region['name']}...")
        
        for year, month in tqdm(months_to_process, desc="Mois", unit="mois"):
            try:
                process_region_month(region, year, month)
                time.sleep(2)  # Pause entre les mois
            except Exception as e:
                print(f"❌ Erreur majeure {region['name']} {year}-{month}: {str(e)}")
                traceback.print_exc()
                time.sleep(30)  # Longue pause après erreur
    
    print("\n✅ Toutes les données mensuelles ont été traitées!")
    
    # Vérification finale
    complete = collection.count_documents({"metadata.status": "complete"})
    partial = collection.count_documents({"metadata.status": "partial"})
    print(f"\n📊 Résultat final:")
    print(f"- Mois complets: {complete}")
    print(f"- Mois partiels: {partial}")
    
    if partial > 0:
        print("\n🔍 Documents partiels à vérifier:")
        for doc in collection.find({"metadata.status": "partial"}).limit(3):
            print(f"{doc['_id']}: {doc.get('metadata', {}).get('errors', 'N/A')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹ Traitement interrompu")
    finally:
        client.close()