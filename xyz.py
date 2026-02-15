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
import random

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
            len(query_tokens.intersection(name_tokens)) >= 1  # At least 1 word matches
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

        # ❌ Skip if nothing matched and not mentioned
        if is_dish_query and not cuisine_hit and not menu_hit:
            if explicitly_mentioned:
                print(f"⚠️ Keeping {name} — No menu match but user asked for dishes from this restaurant")
                place['warning'] = f"I couldn’t find the exact dish names from {name}, but here’s what I know based on reviews or vibe!"
            else:
                print(f"🚫 Skipping {name} — Doesn't match any menu/cuisine terms")
                continue

        # 📏 Radius filter
        if user_lat and user_lon and any(w in query for w in ["near me", "nearby", "around here", "close by"]):
            distance = haversine(user_lat, user_lon, lat, lon)
            print(f"TEST: Distance to {name} = {distance:.2f}km")
            if distance > radius_km:
                print(f"📏 Skipping {name} — {distance:.1f}km > {radius_km}km")
                continue

        # 🕐 Time + ☁️ Weather
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

        # Premium filter
        print(f"TEST: is_premium = {is_premium}, explicitly_mentioned = {explicitly_mentioned}")
        if is_premium_query and not is_premium and not explicitly_mentioned:
            print(f"🚫 Skipping {name} — Not premium")
            continue

        place['is_premium'] = is_premium
        filtered.append(place)
        print(f"✅ TEST: {name} — Added to results ✅")

    return filtered


# 💬 Ask Ollama


def paraphrase_line(text, tries=2):
    try:
        paras = parrot.augment(input_phrase=text, max_return_phrases=tries)
        return paras[0][0] if paras else text
    except:
        return text

def ask_burro(user_query, places, session):
    tone = session.get("tone", "friendly")
    mood = session.get("mood", "neutral")
    location = session.get("location", "Goa")

    if not places:
        return (
            f"Hey! I couldn’t find any places in or around {location} that match your request. "
            f"You could try another mood, cuisine, or nearby area — I’ve got lots of gems to show you when you're ready! 💫"
        )

    header = f"Here are some handpicked spots in {location} based on your mood ({mood}):\n\n"
    body_lines = []

    for i, p in enumerate(places, start=1):
        warning = f"⚠️ {p['warning']}" if 'warning' in p else ""
        premium_tag = "⭐ Premium Dining" if p.get("is_premium") else "❌ Not Premium"
        menu_note = f"\n🍽️ Dish Highlight: {', '.join(p['matched_dishes'])}" if p.get("matched_dishes") else ""

        line = (
            f"{i}. {p['name']} in {p['city']} — {p.get('summary', 'No summary available.')}"
            f"{menu_note}\n⏰ {p.get('time_status', 'Time unknown')} | "
            f"☁️ Weather: {p.get('weather', 'Unknown')} | 💎 {premium_tag} {warning}\n"
            f"📍 Map: {p.get('link', 'N/A')}"
        )

        body_lines.append(paraphrase_line(line))  # Optional: paraphrase each line

    outro = "\n\nLet me know if you want budget options, nightlife vibes, or beachside cafes! 🌴"

    return header + "\n\n".join(body_lines) + outro

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

    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "smollm:1.7b", "prompt": final_prompt, "stream": False}
    )

    return res.json()["response"].strip()

if __name__ == "__main__":
    import sys

    # Optional: Hardcoded user location (can prompt if needed)
    user_lat = 15.2993   # Example: Panaji, Goa
    user_lon = 74.1240

    # Simulate session data
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
        response = ask_burro(user_query, places, session)
        print("\n🤖 Burro:\n" + response + "\n")
    