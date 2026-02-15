# 🐴 Burro  
### AI Chatbot for Discovering Restaurants & Activities in Goa

> Burro is an AI-powered conversational assistant tailored specifically for Goa.  
> It helps users discover restaurants, nightlife, beaches, and activities through intelligent, context-aware recommendations.

---

## 🌴 Overview

Burro is a domain-specific chatbot trained on curated Goan restaurant and activity data.  
Unlike generic travel assistants, Burro understands local context — including cuisine types, popular areas, weather conditions, and time-based availability.

It can answer queries like:

- "Best seafood in North Goa"
- "Romantic beachside dinner spots"
- "Things to do in Baga tonight"
- "Rain-friendly activities today"

---

## 🚀 Features

- 🔎 Smart restaurant discovery
- 🍽 Cuisine-based filtering
- 🎉 Activity & nightlife recommendations
- 🌦 Weather-aware suggestions
- 📍 Geolocation integration
- ⏰ Time-based availability filtering
- 💬 Conversational AI interface
- 📊 Dataset-trained domain intelligence

---

## 🧠 AI Architecture

Burro combines structured data with language model reasoning. The model initially used models from ollama , however they were found to be too heavy for cost effective depolyment on the company server , due to which an api call was preferred .

### Processing Pipeline

User Query  
→ Intent Detection  
→ Dataset Filtering  
→ FAISS based vector querying 
→ Context Enrichment (Time + Weather + Location)  
→ LLM Response Generation  

---

## 🛠 Tech Stack

- **Backend:** Python
- **FAISS based vector querying for accurate results **
- **Model Integration:** LLM API  
- **Geocoding:** Custom geocode module  
- **Weather Integration:** Weather-based contextual suggestions  
- **Time Handling:** Custom time utilities  
- **Environment Management:** `.env`  
- **Dependencies:** `requirements.txt`

---

## 📂 Project Structure

