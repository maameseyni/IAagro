import ee
import time
from pymongo import MongoClient
from datetime import datetime, timedelta
import warnings
from pprint import pprint

# Configuration initiale
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. Initialisation Google Earth Engine
try:
    ee.Initialize(project='ee-imamkebedev')
    print("✅ Connexion à Google Earth Engine établie")
except Exception as e:
    print(f"❌ Erreur GEE: {e}")
    exit()

# 2. Connexion MongoDB
try:
    client = MongoClient(
        "mongodb://localhost:27017",
        #"mongodb+srv://imam:imam@cluster0.ny4c0ui.mongodb.net/",
        serverSelectionTimeoutMS=5000,
        socketTimeoutMS=30000,
        connectTimeoutMS=30000
    )
    client.admin.command('ping')
    db = client["agro_monitoring"]
    collection = db["region_stats"]
    print("✅ Connexion MongoDB établie")
except Exception as e:
    print(f"❌ Erreur MongoDB: {e}")
    exit()

# 3. Paramètres globaux
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=180)  # 6 mois de données
BUFFER_RADIUS = 20000  # 20 km
MAX_CLOUD_COVER = 60  # %

# 4. Collections de données avec priorités
COLLECTIONS = {
    'sentinel': [
        'COPERNICUS/S2_SR_HARMONIZED',
        'COPERNICUS/S2_SR',
        'COPERNICUS/S2_HARMONIZED'
    ],
    'smap': [
        'NASA_USDA/HSL/SMAP_soil_moisture',
        'NASA_USDA/HSL/SMAP10KM_soil_moisture',
        'ECMWF/ERA5_LAND/HOURLY'  # Alternative
    ],
    'modis': [
        'MODIS/061/MOD11A1',
        'MODIS/006/MOD11A1'
    ]
}

# 5. Correction des noms de régions
REGION_CORRECTIONS = {
    "Thiès": "Thiès",
    "Dakar": "Dakar",
    "Saint-Louis": "Saint-Louis",
    "Ziguinchor": "Ziguinchor",
    "Kaolack": "Kaolack",
    "Tambacounda": "Tambacounda",
    "Matam": "Matam",
    "Kolda": "Kolda",
    "Fatick": "Fatick",
    "Louga": "Louga",
    "Kaffrine": "Kaffrine",
    "Diourbel": "Diourbel",
    "Sédhiou": "Sédhiou",
    "Kédougou": "Kédougou",
    "Podor": "Podor"
}

def validate_coordinates(lat, lon):
    """Garantit l'ordre correct [lat, lon]"""
    if abs(lat) > 90 or abs(lon) > 180:
        return lon, lat  # Inverse si nécessaire
    return lat, lon

def get_best_image(collections, region_geom, filter_param=None):
    """Récupère la meilleure image disponible"""
    for collection in collections:
        try:
            coll = ee.ImageCollection(collection)
            
            # Filtres de base
            coll = coll.filterBounds(region_geom).filterDate(START_DATE, END_DATE)
            
            # Filtre supplémentaire si spécifié
            if filter_param:
                coll = coll.filter(ee.Filter.lt(filter_param[0], filter_param[1]))
            
            # Vérification disponibilité
            if coll.size().getInfo() == 0:
                continue
                
            image = coll.first()
            
            # Vérification des bandes pour Sentinel-2
            if 'COPERNICUS/S2' in collection:
                required_bands = ['B2', 'B4', 'B8']
                if not all(b in image.bandNames().getInfo() for b in required_bands):
                    continue
            
            print(f"📡 Utilisation de: {collection}")
            return image
            
        except Exception as e:
            print(f"⚠️ Erreur avec {collection}: {str(e)}")
            continue
    
    print("❌ Aucune image valide trouvée")
    return None

def calculate_vegetation_indices(image):
    """Calcule NDVI et EVI avec gestion d'erreurs"""
    try:
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }
        ).rename('EVI')
        return ndvi, evi
    except Exception as e:
        print(f"❌ Erreur calcul indices: {str(e)}")
        return None, None

def process_region(region):
    """Traite toutes les données pour une région"""
    print(f"\n🌍 Traitement de: {region['name']}")
    
    # Correction et validation
    region_name = REGION_CORRECTIONS.get(region["name"], region["name"])
    lat, lon = validate_coordinates(region["lat"], region["lon"])
    region_geom = ee.Geometry.Point([lon, lat]).buffer(BUFFER_RADIUS)
    
    # 1. Données de végétation (Sentinel-2)
    s2_img = get_best_image(
        COLLECTIONS['sentinel'],
        region_geom,
        ('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER)
    )
    
    vegetation = {
        "ndvi": None, "evi": None,
        "acquisition_date": None,
        "source": None, "cloud_cover": None
    }
    
    if s2_img:
        ndvi, evi = calculate_vegetation_indices(s2_img)
        if ndvi and evi:
            stats = ndvi.addBands(evi).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region_geom,
                scale=100,
                maxPixels=1e9
            ).getInfo()
            
            vegetation.update({
                "ndvi": stats.get('NDVI'),
                "evi": stats.get('EVI'),
                "acquisition_date": s2_img.date().format('YYYY-MM-dd').getInfo(),
                "source": s2_img.get('SPACECRAFT_NAME').getInfo(),
                "cloud_cover": s2_img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
            })
    
    # 2. Humidité du sol (SMAP/ERA5)
    soil_img = get_best_image(COLLECTIONS['smap'], region_geom)
    
    soil = {
        "moisture": None,
        "acquisition_date": None,
        "source": None
    }
    
    if soil_img:
        try:
            band = 'ssm' if 'ssm' in soil_img.bandNames().getInfo() else 'volumetric_soil_water_layer_1'
            moisture = soil_img.select(band).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region_geom,
                scale=10000,
                maxPixels=1e9
            ).getInfo()
            
            if moisture.get(band) is not None:
                soil.update({
                    "moisture": moisture[band],
                    "acquisition_date": soil_img.date().format('YYYY-MM-dd').getInfo(),
                    "source": soil_img.get('system:index').getInfo().split('_')[0]
                })
        except Exception as e:
            print(f"❌ Erreur humidité: {str(e)}")
    
    # 3. Température de surface (MODIS)
    modis_img = get_best_image(COLLECTIONS['modis'], region_geom)
    
    temperature = {
        "surface": None,
        "acquisition_date": None,
        "source": None
    }
    
    if modis_img:
        lst = modis_img.select('LST_Day_1km').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region_geom,
            scale=1000,
            maxPixels=1e9
        ).getInfo()
        
        if lst.get('LST_Day_1km') is not None:
            temperature.update({
                "surface": lst['LST_Day_1km'] * 0.02 - 273.15,  # Conversion en °C
                "acquisition_date": modis_img.date().format('YYYY-MM-dd').getInfo(),
                "source": "MODIS"
            })
    
    # Construction du document final
    doc = {
        "region": region_name,
        "location": {
            "type": "Point",
            "coordinates": [lon, lat]
        },
        "vegetation": vegetation,
        "soil": soil,
        "temperature": temperature,
        "metadata": {
            "processed_at": datetime.now().isoformat(),
            "algorithm_version": "1.2"
        }
    }
    
    return doc

def main():
    """Point d'entrée principal"""
    regions = [
        {"name": "Dakar", "lat": 14.7167, "lon": -17.4677},
        {"name": "Thids", "lat": 14.8136, "lon": -16.9622},
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
    
    for region in regions:
        start_time = time.time()
        
        try:
            # Traitement de la région
            data = process_region(region)
            
            # Vérification qualité
            if all([data["soil"]["moisture"] is None,
                   data["vegetation"]["ndvi"] is None,
                   data["temperature"]["surface"] is None]):
                print("⚠️ Aucune donnée valide trouvée")
                continue
            
            # Sauvegarde MongoDB
            collection.update_one(
                {"region": data["region"]},
                {"$set": data},
                upsert=True
            )
            
            # Affichage synthétique
            print(f"✅ Données sauvegardées - NDVI: {data['vegetation']['ndvi']:.2f} | "
                  f"Humidité: {data['soil']['moisture'] or 'NA'} | "
                  f"Temp: {data['temperature']['surface'] or 'NA'}°C")
            
        except Exception as e:
            print(f"❌ ERREUR: {str(e)}")
        finally:
            print(f"⏱ Temps: {time.time() - start_time:.2f}s")
            time.sleep(1)  # Pause entre régions
    
    client.close()
    print("\n✨ Traitement terminé pour toutes les régions")

if __name__ == "__main__":
    main()
