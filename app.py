from flask import Flask, request, jsonify, render_template
from test2 import recommend_places, ask_gemini
from utils.geocode import reverse_geocode

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("message")
    user_lat = data.get("latitude")
    user_lon = data.get("longitude")
    radius_km = float(data.get("radius", 7))

    if user_lat and user_lon:
        location = reverse_geocode(user_lat, user_lon)
    else:
        location = "Goa"

    session = {
        "location": location
    }

    try:
        places = recommend_places(user_query, user_lat, user_lon, radius_km)
        response = ask_gemini(user_query, places, session)
        return jsonify({"reply": response})
    except Exception as e:
        print("💥 Error:", e)
        return jsonify({
            "reply": "Oops! Something went wrong on Burro's side 🐴. Please try again later."
        })

if __name__ == "__main__":
    app.run(debug=True)
