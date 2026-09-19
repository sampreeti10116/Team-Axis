from flask import Flask, render_template, request, jsonify
from ml_engine import engine

app = Flask(__name__, template_folder="templates", static_folder="static")

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
        algo_name = data.get("algorithm", "Random Forest")

        res = engine.predict_yield(
            crop, location, season, irrigation, field_area,
            soil_ph, moisture, n, p, k, organic_matter, algo_name
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
    return jsonify({"success": True, "reply": reply})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
