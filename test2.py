import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import re  # 🔥 Added for punctuation removal
from utils.weather import get_current_weather
from utils.time_utils import is_place_open_now
from math import radians, sin, cos, sqrt, atan2
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
load_dotenv()
import os
import json
from datetime import datetime
from pathlib import Path

# 📌 NEW: GeminiKeyManager for API rotation & usage tracking
class GeminiKeyManager:
    def __init__(self, keys_env_var="GEMINI_API_KEYS", usage_file=".gemini_usage.json", daily_limit=2):
        keys_str = os.getenv(keys_env_var, "")
        self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not self.keys:
            raise ValueError("No API keys found in environment.")

        self.usage_file = Path(usage_file)
        self.daily_limit = daily_limit
        self.state = self._load_state()

        # Reset daily usage if date changed
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state["last_date"] != today:
            self.state["last_date"] = today
            self.state["usage"] = {k: 0 for k in self.keys}
            self._save_state()

    def _load_state(self):
        if self.usage_file.exists():
            return json.loads(self.usage_file.read_text())
        return {
            "last_date": "",
            "current_index": 0,
            "usage": {k: 0 for k in self.keys}
        }

    def _save_state(self):
        self.usage_file.write_text(json.dumps(self.state))

    def get_key(self):
        return self.keys[self.state["current_index"]]

    def increment_usage(self):
        """Increment usage and rotate if limit exceeded."""
        key = self.get_key()
        self.state["usage"][key] += 1
        print(f"📊 Gemini Key Usage: {key} → {self.state['usage'][key]} calls today")
        if self.state["usage"][key] >= self.daily_limit:
            print(f"⚠️ Limit reached for key {key}. Rotating to next key.")
            self._rotate_key()
        self._save_state()

    def _rotate_key(self):
        self.state["current_index"] = (self.state["current_index"] + 1) % len(self.keys)
        print(f"🔄 Switched to key index {self.state['current_index']}")

# 📍 Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Load model + FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("index/places.index")
with open("index/places_meta.pkl", "rb") as f:
    metadatas = pickle.load(f)

# 🔍 Semantic Search
def search_places(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [metadatas[i] for i in indices[0]]

# 🔍 Main recommendation logic
def recommend_places(user_query, user_lat=None, user_lon=None, radius_km=7):
    query = re.sub(r'[^\w\s]', '', user_query.strip().lower())
    print(f"[User Query] {query}")

    is_premium_query = any(x in query for x in ["premium", "fine dining", "luxury", "rich vibe", "expensive", "top-tier"])
    is_dish_query = any(word in query for word in ["serve", "get", "have", "eat", "dish", "try", "offer", "dishes", "menu"])
    dish_keywords = [word for word in query.split() if len(word) > 3]

    print(f"TEST: is_premium_query = {is_premium_query}, is_dish_query = {is_dish_query}, dish_keywords = {dish_keywords}")

    raw_results = search_places(query)
    filtered = []

    print("\n📌 RAW RESULTS FROM FAISS + fallback:")
    for r in raw_results:
        print("—", r["name"])

    for place in raw_results:
        name = place.get('name', '').strip()
        name_lower = name.lower()
        lat = place.get('latitude')
        lon = place.get('longitude')
        timings = place.get('timings', [])
        outdoor = place.get('outdoor_seating', False)

        if "premium_added" not in place:
            print(f"❗TEST FAIL: {name} — 'premium_added' key is missing!")
        else:
            print(f"✅ TEST: {name} — premium_added = {place['premium_added']}")

        is_premium = place.get("premium_added", False)
        cuisines = place.get("cuisines", [])
        menu_items = place.get("menu", [])

        # 🔍 Explicit match logic
        variants = [name_lower]
        if '||' in name:
            variants.append(name.split('||')[0].strip().lower())

        query_tokens = set(query.split())
        name_tokens = set()
        for variant in variants:
            name_tokens.update(variant.lower().split())

        explicitly_mentioned = (
            any(v in query for v in variants) or
            any(fuzz.partial_ratio(query, v) > 85 for v in variants) or
            len(query_tokens.intersection(name_tokens)) >= 1
        )

        print(f"TEST: explicitly_mentioned = {explicitly_mentioned}")

        # 🔍 Cuisine/Dish match
        cuisine_hit = any(c.lower() in query for c in cuisines)
        matched_dishes = []
        for word in dish_keywords:
            for item in menu_items:
                if fuzz.partial_ratio(word.lower(), item.lower()) > 80:
                    matched_dishes.append(item)
                    break

        menu_hit = len(matched_dishes) > 0

        # 🔁 Full fallback if explicitly mentioned
        if is_dish_query and not menu_hit and explicitly_mentioned and menu_items:
            matched_dishes = menu_items
            print(f"⚠️ Full menu returned for {name} due to explicit mention in dish query")
            menu_hit = True

        place["matched_dishes"] = matched_dishes
        print(f"TEST: cuisine_hit = {cuisine_hit}, menu_hit = {menu_hit}, matched_dishes = {matched_dishes}")

        if is_dish_query and not cuisine_hit and not menu_hit:
            if explicitly_mentioned:
                print(f"⚠️ Keeping {name} — No menu match but user asked for dishes from this restaurant")
                place['warning'] = f"I couldn’t find the exact dish names from {name}, but here’s what I know based on reviews or vibe!"
            else:
                print(f"🚫 Skipping {name} — Doesn't match any menu/cuisine terms")
                continue

        if user_lat and user_lon and any(w in query for w in ["near me", "nearby", "around here", "close by"]):
            distance = haversine(user_lat, user_lon, lat, lon)
            print(f"TEST: Distance to {name} = {distance:.2f}km")
            if distance > radius_km:
                print(f"📏 Skipping {name} — {distance:.1f}km > {radius_km}km")
                continue

        weather = get_current_weather(lat, lon)
        place['weather'] = weather
        is_open, time_msg = is_place_open_now(timings)
        place['time_status'] = time_msg
        print(f"TEST: is_open = {is_open}, time_status = {time_msg}, weather = {weather}")

        if not is_open:
            if explicitly_mentioned or (is_premium_query and is_premium):
                print(f"⚠️ Keeping {name} — Closed but acceptable")
                place['warning'] = f"{name} is currently closed. {time_msg}"
            else:
                print(f"⛔ Skipping {name} — Closed")
                place['warning'] = f"{name} is closed now. {time_msg}"
                continue

        if weather == "rainy" and outdoor and not explicitly_mentioned:
            print(f"⛔ Skipping {name} — Outdoor & Raining")
            place['warning'] = f"⚠️ Rain alert: {name} has outdoor seating and it’s currently raining."
            continue
        elif weather == "rainy" and outdoor and explicitly_mentioned:
            print(f"⚠️ Keeping {name} — Raining but mentioned")
            place['warning'] = f"It’s raining there now 🌧️ and {name} has outdoor seating."

        print(f"TEST: is_premium = {is_premium}, explicitly_mentioned = {explicitly_mentioned}")
        if is_premium_query and not is_premium and not explicitly_mentioned:
            print(f"🚫 Skipping {name} — Not premium")
            continue

        place['is_premium'] = is_premium
        filtered.append(place)
        print(f"✅ TEST: {name} — Added to results ✅")

    return filtered

import google.generativeai as genai

# 📌 Create key manager instance
key_manager = GeminiKeyManager(daily_limit=2)

def ask_gemini(user_query, places, session):
    tone = session.get("tone", "friendly")
    mood = session.get("mood", "neutral")
    location = session.get("location", "Goa")

    if not places:
        return (
            f"Hey! I couldn’t find any places in or around {location} that match your request. "
            f"You could try another mood, cuisine, or nearby area — I’ve got lots of gems to show you when you're ready! 💫"
        )

    system_prompt = (
        f"You are Burro, a helpful local travel assistant for Goa. "
        f"Speak in a {tone} tone and adapt to the user's mood: {mood}. "
        f"The user is currently in {location}. "
        f"Only recommend places from the list below. "
        f"NEVER invent or mention any places not in this list. "
        f"If the user's query seems unrelated or irrelevant, respond politely and redirect them to ask about food, places, or activities in Goa. "
        f"If describing a place as premium, ONLY do so if 'Premium Added' is true."
    )

    formatted_places = []
    for i, p in enumerate(places, start=1):
        warning = f"⚠️ {p['warning']}" if 'warning' in p else ""
        premium_tag = "⭐ Premium Dining" if p.get("is_premium") else "❌ Not Premium"
        menu_note = ""
        if p.get("matched_dishes"):
            menu_note = f"\n🍽️ Dish Highlight: {', '.join(p['matched_dishes'])}"

        formatted_places.append(
            f"{i}. {p['name']} in {p['city']}: {p.get('summary', 'No summary available.')}"
            f"{menu_note}"
            f"\nCurrently: {p.get('time_status', 'Time unknown')} | Weather: {p.get('weather', 'Unknown')} | "
            f"💎 Premium: {premium_tag} {warning}"
            f"\nMap: {p.get('link', 'N/A')}"
        )

    final_prompt = (
        f"{system_prompt}\n\n"
        f"ONLY use the following places to reply. Do NOT make up names or suggestions.\n\n"
        f"User: {user_query}\n\n"
        f"Places:\n" + "\n\n".join(formatted_places)
    )

    genai.configure(api_key=key_manager.get_key())

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(final_prompt)
    finally:
        key_manager.increment_usage()

    return response.text.strip()

if __name__ == "__main__":
    import sys

    user_lat = 15.2993
    user_lon = 74.1240

    session = {
        "tone": "friendly",
        "mood": "neutral",
        "location": "Goa"
    }

    print("👋 Welcome to Burro — Your Goa Guide!")
    print("Type your question (type 'exit' or 'quit' to stop):\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("👋 Bye! Have a great day in Goa!")
            sys.exit()

        places = recommend_places(user_query, user_lat=user_lat, user_lon=user_lon)
        print("\n📦 Final Filtered Places sent to Gemini:")
        import json
        print(json.dumps(places, indent=2))

        response = ask_gemini(user_query, places, session)
        print("\n🤖 Burro:\n" + response + "\n")
