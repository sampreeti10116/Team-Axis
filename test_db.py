import os
import unittest
from database import init_db, save_prediction, save_soil_analysis, save_chat_log, get_db_stats, db_session, PredictionRecord, SoilAnalysisRecord, ChatLogRecord

class TestDatabaseIntegration(unittest.TestCase):
    def setUp(self):
        init_db()

    def test_save_prediction(self):
        pred_id = save_prediction(
            crop="Cotton", location="Gujarat", season="Kharif", irrigation="Drip",
            field_area=12.5, soil_ph=7.2, moisture=50.0, n=110, p=55, k=65, om=2.8,
            algo="Gradient Boosting", predicted_yield=3.85, total_harvest=48.125
        )
        self.assertIsNotNone(pred_id)
        
        session = db_session()
        rec = session.query(PredictionRecord).filter_by(id=pred_id).first()
        self.assertEqual(rec.crop, "Cotton")
        self.assertEqual(rec.location, "Gujarat")
        self.assertEqual(rec.predicted_yield, 3.85)
        session.close()

    def test_save_soil_analysis(self):
        soil_id = save_soil_analysis(
            ph=6.8, moisture=48.0, n=95, p=50, k=60, om=2.4, crop="Sugarcane",
            soil_status="Optimal", recommendation_summary="Apply balanced organic compost."
        )
        self.assertIsNotNone(soil_id)

        session = db_session()
        rec = session.query(SoilAnalysisRecord).filter_by(id=soil_id).first()
        self.assertEqual(rec.intended_crop, "Sugarcane")
        self.assertEqual(rec.soil_status, "Optimal")
        session.close()

    def test_save_chat_log(self):
        chat_id = save_chat_log(
            user_message="How much nitrogen does rice need?",
            ai_reply="Rice requires approximately 100-120 kg/ha of Nitrogen split into 3 applications."
        )
        self.assertIsNotNone(chat_id)

        session = db_session()
        rec = session.query(ChatLogRecord).filter_by(id=chat_id).first()
        self.assertIn("Nitrogen", rec.ai_reply)
        session.close()

    def test_get_db_stats(self):
        stats = get_db_stats()
        self.assertIn("database_type", stats)
        self.assertGreaterEqual(stats["predictions_count"], 1)

if __name__ == "__main__":
    unittest.main()
