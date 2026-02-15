import requests

def reverse_geocode(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {"User-Agent": "BurroBot/1.0"}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        address = data.get("address", {})
        return address.get("city") or address.get("town") or address.get("village") or "your area"
    except Exception as e:
        print("Reverse geocoding failed:", e)
        return "your area"
