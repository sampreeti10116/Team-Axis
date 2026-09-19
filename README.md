# AgriYield AI 🌾
> **Smart Crop Yield Prediction & Farm Intelligence Platform**

AgriYield AI is an enterprise-grade AI/ML software platform designed to empower individual farmers, agronomists, and agricultural cooperatives with data-driven decision-making insights. By combining soil analytics, weather patterns, historical crop performance, market price trends, government scheme intelligence, and machine learning models, AgriYield AI optimizes farm productivity and crop yield accuracy.

---

## 🌟 Key Features

- 📈 **AI/ML Crop Yield Prediction**: Features a high-precision Machine Learning engine trained on 10,000+ agricultural data rows. Includes **Gradient Boosting** ($R^2 = 0.9508$, $95.1\%$ accuracy), **Random Forest** ($R^2 = 0.899$), **Decision Tree**, and **Linear Regression**.
- 🌱 **Soil & Nutrient Analysis**: Evaluates Soil pH, Moisture, Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$), and Organic Matter ($OM\%$) to deliver tailored fertilizer dosage advice and crop suitability matrix.
- 🌦️ **Weather & Climate Intelligence**: Live weather forecasts, precipitation trends, temperature monitoring, and actionable irrigation risk advisories.
- 📊 **Historical Crop Data Trends**: Interactive multi-year yield trajectory visualization built with Chart.js.
- 💰 **Market Intelligence & Price Trends**: Real-time Minimum Support Price (MSP), local mandi price tracking, and bullish/bearish market sentiment.
- 🏛️ **Government Schemes & Subsidies**: Instant eligibility breakdown and application links for key agricultural schemes (*PM-KISAN*, *PMFBY*, *Soil Health Card*).
- 🤖 **AI Platform Guide (Mascot Assistant)**: Context-aware interactive assistant providing real-time agronomic guidance and troubleshooting.
- 👔 **Dual Persona View**: One-click toggle between **Farmer View** (glanceable, clear summary cards) and **Agronomist Command Center** (multi-farm portfolio metrics, soil risk flags, and model $R^2$ benchmarks).
- 🗄️ **PostgreSQL Database Storage**: Production-ready data persistence using SQLAlchemy ORM and `psycopg2-binary` with zero-downtime local SQLite fallback.

---

## 🛠️ Technology Stack

| Layer | Technologies Used |
|---|---|
| **Frontend** | HTML5, Vanilla CSS3 (Organic & Earthy High Contrast Design System), JavaScript (ES6+), SVG Line-Art Icons, Chart.js |
| **Backend** | Python 3.11+, Flask REST API, Gunicorn / Development WSGI |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy, Joblib |
| **Database** | PostgreSQL, SQLAlchemy ORM, Psycopg2, SQLite (Local Fallback) |

---

## 📂 Project Structure

```text
Team-Axis/
├── app.py                     # Flask application & REST API routing
├── ml_engine.py               # Scikit-Learn ML training, prediction & agronomic logic
├── database.py                 # PostgreSQL connection & SQLAlchemy ORM models
├── test_db.py                 # Database unit test suite
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
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

### 2. Clone the Repository
```bash
git clone https://github.com/sampreeti10116/Team-Axis.git
cd Team-Axis
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Optional for PostgreSQL)
Copy `.env.example` to `.env` and set your PostgreSQL connection string:
```bash
cp .env.example .env
```
In `.env`:
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/agriyield_db
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
| `GET`  | `/api/weather` | Fetch weather forecast and irrigation advisories by location |
| `GET`  | `/api/historical-trends` | Get historical harvest trajectories and yield trends |
| `GET`  | `/api/market-intelligence` | Fetch MSP, market prices, and crop trade sentiment |
| `GET`  | `/api/schemes` | Retrieve active government agricultural schemes |
| `POST` | `/api/chat-guide` | Interact with the AI agronomic assistant mascot |
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
