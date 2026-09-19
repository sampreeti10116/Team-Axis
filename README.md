# AgriYield AI 🌾
> **Smart Crop Yield Prediction & Farm Intelligence Platform**

AgriYield AI is an enterprise-grade AI/ML software platform designed to empower individual farmers, agronomists, and agricultural cooperatives with data-driven decision-making insights. By combining soil analytics, weather patterns, historical crop performance, market price trends, government scheme intelligence, machine learning models, and real-time Gemini Generative AI, AgriYield AI optimizes farm productivity and crop yield accuracy.

---

## 🌟 Key Features

- 📈 **AI/ML Crop Yield Prediction**: Features a high-precision Machine Learning engine trained on 10,000+ agricultural data rows. Includes **Gradient Boosting** ($R^2 = 0.9508$, $95.1\%$ accuracy), **Random Forest** ($R^2 = 0.899$), **Decision Tree**, and **Linear Regression**. Yield predictions adjust dynamically based on regional climate profile and soil factors.
- 🌾 **Automatic Crop Season Classification**: Automatically classifies crops into **Rabi**, **Kharif**, or **Zaid** sowing cycles based on selected crop species and region.
- 🌱 **Soil & Nutrient Analysis**: Evaluates Soil pH, Moisture, Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$), and Organic Matter ($OM\%$) to deliver tailored fertilizer dosage advice and soil health scores (0–100).
- 🌦️ **Dynamic Location-Aware Weather**: Live location-based weather metrics (temperature, humidity, evapotranspiration, rainfall forecast) that update dynamically for any selected state or district in India.
- 📊 **Dynamic Historical Yield Analytics**: Interactive multi-year harvest yield trajectories (2018–2025) built with Chart.js, dynamically reflecting selected crop and regional benchmarks.
- 💰 **Market Intelligence & Price Trends**: Real-time APMC mandi prices, distance-based market price comparisons, and post-harvest storage guidance.
- 🏛️ **Government Schemes & Subsidies**: Instant eligibility breakdown and application links for key agricultural schemes (*PMFBY*, *Soil Health Card*, *PMKSY*, *SMAM*).
- 🤖 **Dynamic Gemini Generative AI Agent**: Real-time conversational AI platform guide powered by Google's **Gemini AI** (`gemini-flash-latest`), providing instant agronomic advice, pest management tips, and platform assistance with quick prompt chips.
- 👔 **Dual Persona View**: One-click toggle between **Farmer View** (glanceable, clear summary cards) and **Agronomist Command Center** (multi-farm portfolio metrics, soil risk flags, and model $R^2$ benchmarks).
- 🗄️ **PostgreSQL Database Storage**: Production-ready data persistence using SQLAlchemy ORM and `psycopg2-binary` with zero-downtime local SQLite fallback.

---

## 🛠️ Technology Stack

| Layer | Technologies Used |
|---|---|
| **Frontend** | HTML5, Vanilla CSS3 (Organic & Earthy High Contrast Design System), JavaScript (ES6+), SVG Line-Art Icons, Chart.js |
| **Backend** | Python 3.11+, Flask REST API, Gunicorn / Development WSGI |
| **Generative AI Agent** | Google Gemini Generative AI API (`gemini-flash-latest`), REST Integration |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy, Joblib |
| **Database** | PostgreSQL, SQLAlchemy ORM, Psycopg2, SQLite (Local Fallback) |

---

## 📂 Project Structure

```text
Team-Axis/
├── app.py                     # Flask application & REST API routing
├── ml_engine.py               # Scikit-Learn ML engine & Gemini AI Agent handler
├── database.py                # PostgreSQL connection & SQLAlchemy ORM models
├── test_db.py                 # Database integration unit test suite
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── .env                       # Local environment secrets (ignored by Git)
├── data/
│   └── processed/
│       └── final_features.csv # Processed dataset for model training
├── models/                    # Serialized model binaries (.pkl)
├── static/
│   ├── css/
│   │   └── style.css          # Design tokens & high-contrast styling
│   └── js/
│       └── main.js            # Client UI state & API interaction logic
└── templates/
    └── index.html             # Single-page dynamic dashboard UI
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- Git
- PostgreSQL (Optional for local development; automatic fallback to SQLite is provided)
- Google Gemini API Key (Optional; fallback knowledge engine is provided if unconfigured)

### 2. Clone the Repository
```bash
git clone https://github.com/sampreeti10116/Team-Axis.git
cd Team-Axis
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and set your credentials:
```bash
cp .env.example .env
```
In `.env`:
```env
# Database Configuration (Optional - defaults to SQLite)
DATABASE_URL=postgresql://postgres:password@localhost:5432/agriyield_db

# Gemini Generative AI Key (For Dynamic AI Agent Assistant)
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Predict crop yield (Tonnes/Ha) based on soil, weather, & field parameters |
| `POST` | `/api/soil-analysis` | Analyze soil nutrient deficiencies and get fertilizer dosage advice |
| `GET`  | `/api/weather` | Fetch location-dependent weather forecast & irrigation advisories |
| `GET`  | `/api/historical-trends` | Get crop and location-specific historical harvest trajectories |
| `GET`  | `/api/market-intelligence` | Fetch APMC mandi prices, nearby market price comparisons, & tips |
| `GET`  | `/api/schemes` | Retrieve verified active government agricultural schemes |
| `POST` | `/api/chat-guide` | Interact with the dynamic Gemini Generative AI Agronomic Assistant |
| `GET`  | `/api/db-status` | Monitor active database connection engine and record counts |

---

## 🧪 Testing

To run the automated database integration and persistence unit test suite:
```bash
python test_db.py
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
