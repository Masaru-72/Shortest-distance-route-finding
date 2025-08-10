import os
import csv
import numpy as np
import requests

def load_stop_data(filepath="data/Location/location.csv"):
    """Loads stop data from a CSV file."""
    stops = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The data file '{filepath}' was not found. Please make sure it is in the same directory.")
    
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Updated the check to ensure avg_passenger column exists
            if 'name' not in row or 'lat' not in row or 'lng' not in row or 'avg_passenger' not in row:
                raise KeyError(f"The CSV file must have 'name', 'lat', 'lng', and 'avg_passenger' columns.")
            
            # Added 'avg_passenger' to the dictionary, converting it to an integer
            stops.append({
                "id": i + 1, 
                "name": row['name'], 
                "lat": float(row['lat']), 
                "lng": float(row['lng']),
                "avg_passenger": float(row['avg_passenger'])
            })
    return stops
