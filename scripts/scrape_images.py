import requests
import time
import math
import os
import json
from PIL import Image
from io import BytesIO

# List of neighborhoods/landmarks in Toronto
locations = [
    "Financial District, Toronto, ON",
    # "Entertainment District, Toronto, ON",
    # "Chinatown, Toronto, ON",
    # "Kensington Market, Toronto, ON",
    # "St. Lawrence Market, Toronto, ON",
    # "Distillery District, Toronto, ON",
    # "Waterfront, Toronto, ON",
    # "University of Toronto Campus, Toronto, ON",
    # Add the rest of the 64 locations
]

# Function to get coordinates using Nominatim API from OpenStreetMap
def get_coordinates(location):
    url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 200:
        data = response.json()
        if data:
            coords = data[0]
            return float(coords['lat']), float(coords['lon'])
        else:
            print(f"No results found for {location}")
            return None
    else:
        print(f"Error fetching data for {location}: {response.status_code}")
        return None

# Function to convert latitude/longitude to tile numbers for a specific zoom level
def lat_lon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

# Updated function to download a map tile from OpenTopoMap
def download_topo_tile(location_name, lat, lon, zoom=13):
    xtile, ytile = lat_lon_to_tile(lat, lon, zoom)
    url = f"https://a.tile.opentopomap.org/{zoom}/{xtile}/{ytile}.png"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        folder = "topo_tiles"
        os.makedirs(folder, exist_ok=True)
        img = Image.open(BytesIO(response.content))
        img.save(f"{folder}/{location_name.replace(' ', '_')}.png")
        print(f"Downloaded map tile for {location_name} at zoom level {zoom}")
    else:
        print(f"Failed to download tile for {location_name}: Status code {response.status_code}")

# Query and save all locations with their coordinates
location_coords = {}

for location in locations:
    lat_lng = get_coordinates(location)
    if lat_lng:
        location_coords[location] = lat_lng
        print(f"Coordinates for {location}: {lat_lng}")
        # Download the map tile for the location
        download_topo_tile(location, lat_lng[0], lat_lng[1])
        time.sleep(3)  # Increased delay to be more respectful of the server

# Save coordinates to a JSON file
with open("toronto_locations_topo.json", "w") as json_file:
    json.dump(location_coords, json_file, indent=4)
    print("Saved all coordinates to toronto_locations_topo.json")
