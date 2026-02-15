import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
DEFAULT_LAT = os.getenv("DEFAULT_LAT", 15.2993)  # Goa lat
DEFAULT_LON = os.getenv("DEFAULT_LON", 74.1240)  # Goa lon

def get_current_weather(lat=DEFAULT_LAT, lon=DEFAULT_LON):
    if not API_KEY:
        return "unknown"

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        )
        response = requests.get(url)
        data = response.json()
        weather = data['weather'][0]['main'].lower()

        if "rain" in weather:
            return "rainy"
        elif "cloud" in weather:
            return "cloudy"
        elif "clear" in weather:
            return "sunny"
        else:
            return weather
    except:
        return "unknown"
