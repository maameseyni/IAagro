import requests
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import time

# 1. Configuration complète des indicateurs et régions
CONFIG = {
    "api_url": "https://api.worldbank.org/v2/country/SN/indicator/",
    "indicators": {
        "Prix_riz": "FP.CPI.TOTL.FD.ZG",  # Inflation des prix alimentaires
        "Prix_mil": "FP.CPI.TOTL",        # Indice des prix à la consommation
        "Prix_maïs": "AG.PRD.CROP.XD",    # Production céréalière (proxy)
        "Prix_huile": "FP.CPI.FOOD.ZG",   # Inflation alimentaire
        "Prix_sucre": "AG.PRD.SUGR.ZS"    # Prix du sucre
    },
    "regions": {
        "Dakar": "SN-DKR",
        "Thiès": "SN-THS", 
        "Saint-Louis": "SN-STL",
        "Diourbel": "SN-DBR",
        "Ziguinchor": "SN-ZIG"
    }
}

# 2. Initialisation MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["food_security_db"]
prices_col = db["region_food_prices"]

# 3. Fonction optimisée de récupération
def fetch_wb_data(indicator, region_code=None):
    base_url = f"{CONFIG['api_url']}{indicator}" if not region_code \
              else f"https://api.worldbank.org/v2/region/{region_code}/indicator/{indicator}"
    
    params = {
        "format": "json",
        "per_page": 500,
        "date": "2010:2025",
        "frequency": "M"  # Données mensuelles
    }

    try:
        response = requests.get(base_url, params=params, timeout=45)
        response.raise_for_status()
        data = response.json()
        return data[1] if len(data) > 1 else None
        
    except Exception as e:
        print(f"Erreur {indicator} ({region_code}): {str(e)}")
        return None

# 4. Transformation des données
def process_data(raw_data, product, region=None):
    if not raw_data:
        return None
        
    df = pd.DataFrame(raw_data)[['date', 'value']]
    df = df[df['value'].notna()]
    
    df['date'] = pd.to_datetime(df['date'] + '-01')  # Format YYYY-MM-DD
    df['value'] = pd.to_numeric(df['value'])
    
    df.insert(0, 'product', product)
    if region:
        df.insert(1, 'region', region)
    
    df['last_update'] = datetime.now()
    return df.to_dict('records')

# 5. Pipeline complet
def main():
    print("Début de l'importation...\n")
    
    # Données nationales
    for product, indicator in CONFIG['indicators'].items():
        print(f"Traitement national: {product}...")
        data = fetch_wb_data(indicator)
        records = process_data(data, product)
        
        if records:
            prices_col.insert_many(records)
            print(f"→ {len(records)} enregistrements")
        time.sleep(1.5)
    
    # Données régionales
    for region, code in CONFIG['regions'].items():
        print(f"\nTraitement région {region}:")
        
        for product, indicator in CONFIG['indicators'].items():
            print(f"- {product}...", end=' ')
            data = fetch_wb_data(indicator, code)
            records = process_data(data, product, region)
            
            if records:
                prices_col.insert_many(records)
                print(f"{len(records)} mois")
            else:
                print("x")
            time.sleep(2)
    
    print("\nMise à jour terminée!")

if __name__ == "__main__":
    main()