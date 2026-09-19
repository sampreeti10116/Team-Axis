from flask import Flask, render_template, request, jsonify
from ml_engine import engine
from database import init_db, save_prediction, save_soil_analysis, save_chat_log, get_db_stats

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize PostgreSQL / SQLAlchemy database schema on application start
init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        crop = data.get("crop", "Wheat")
        location = data.get("location", "Punjab, India")
        season = data.get("season", "Kharif")
        irrigation = data.get("irrigation", "Drip")
        field_area = float(data.get("field_area", 5))
        soil_ph = float(data.get("soil_ph", 6.5))
        moisture = float(data.get("moisture", 45))
        n = float(data.get("n", 100))
        p = float(data.get("p", 60))
        k = float(data.get("k", 60))
        organic_matter = float(data.get("organic_matter", 2.5))
        algo_name = data.get("algorithm") or "Gradient Boosting"

        res = engine.predict_yield(
            crop, location, season, irrigation, field_area,
            soil_ph, moisture, n, p, k, organic_matter, algo_name
        )

        # Persist prediction in PostgreSQL database
        save_prediction(
            crop=crop, location=location, season=season, irrigation=irrigation,
            field_area=field_area, soil_ph=soil_ph, moisture=moisture,
            n=n, p=p, k=k, om=organic_matter, algo=res.get("algorithm_used", algo_name),
            predicted_yield=res.get("predicted_yield_tonnes_per_hectare", 0),
            total_harvest=res.get("total_harvest_tonnes", 0)
        )

        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/soil-analysis", methods=["POST"])
def api_soil_analysis():
    try:
        data = request.get_json(force=True)
        ph = float(data.get("soil_ph", 6.5))
        moisture = float(data.get("moisture", 45))
        n = float(data.get("n", 100))
        p = float(data.get("p", 60))
        k = float(data.get("k", 60))
        om = float(data.get("organic_matter", 2.5))
        crop = data.get("intended_crop", "Wheat")

        res = engine.analyze_soil(ph, moisture, n, p, k, om, crop)

        # Persist soil test record in PostgreSQL database
        save_soil_analysis(
            ph=ph, moisture=moisture, n=n, p=p, k=k, om=om, crop=crop,
            soil_status=res.get("status", "Normal"),
            recommendation_summary=str(res.get("recommendations", []))
        )

        return jsonify({"success": True, "data": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/weather", methods=["GET"])
def api_weather():
    location = request.args.get("location", "Punjab, India")
    res = engine.get_weather_forecast(location)
    return jsonify({"success": True, "data": res})

@app.route("/api/historical-trends", methods=["GET"])
def api_history():
    crop = request.args.get("crop", "Wheat")
    location = request.args.get("location", "Punjab, India")
    res = engine.get_historical_trends(crop, location)
    return jsonify({"success": True, "data": res})

@app.route("/api/market-intelligence", methods=["GET"])
def api_market():
    crop = request.args.get("crop", "Wheat")
    res = engine.get_market_intelligence(crop)
    return jsonify({"success": True, "data": res})

@app.route("/api/schemes", methods=["GET"])
def api_schemes():
    res = engine.get_schemes()
    return jsonify({"success": True, "data": res})

@app.route("/api/chat-guide", methods=["POST"])
def api_chat_guide():
    data = request.get_json(force=True)
    user_msg = data.get("message", "")
    reply = engine.answer_guide_query(user_msg)

    # Persist chat interaction in PostgreSQL database
    save_chat_log(user_message=user_msg, ai_reply=reply)

    return jsonify({"success": True, "reply": reply})

@app.route("/api/db-status", methods=["GET"])
def api_db_status():
    stats = get_db_stats()
    return jsonify({"success": True, "data": stats})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
