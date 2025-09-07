import requests
from pymongo import MongoClient
from datetime import datetime

# Configuration MongoDB
MONGO_URI = "mongodb://localhost:27017"
#MONGO_URI = "mongodb+srv://imam:imam@cluster0.ny4c0ui.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["weather_data"]
collection = db["regional_weather"]

# Configuration de l'API OpenWeatherMap
API_URL = "http://api.openweathermap.org/data/2.5/forecast"
API_KEY = "af38ec2a7acabc2b48f76b881a5dd17c"  # Remplacez par votre clé API

# Liste des régions avec leurs coordonnées
regions = [
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

def fetch_weather_data(region):
    """Récupère les données météo pour une région spécifique."""
    params = {
        "lat": region["lat"],
        "lon": region["lon"],
        "appid": API_KEY,
        "lang": "fr",
        "units": "metric"
    }
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erreur pour la région {region['name']}: {e}")
        return None

def save_to_mongodb(region, data):
    """Insère les données météo dans MongoDB."""
    if not data:
        print(f"Aucune donnée pour {region['name']}")
        return
    
    for item in data["list"]:
        document = {
            "region": region["name"],
            "datetime": datetime.fromtimestamp(item["dt"]),
            "temperature": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "weather": item["weather"][0]["description"],
            "wind_speed": item["wind"]["speed"]
        }
        collection.insert_one(document)
        print(f"Données insérées pour {region['name']}: {document}")

def main():
    for region in regions:
        print(f"Récupération des données pour {region['name']}...")
        weather_data = fetch_weather_data(region)
        save_to_mongodb(region, weather_data)
    print("Terminé pour toutes les régions.")

if __name__ == "__main__":
    main()