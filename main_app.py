from flask import Flask, send_file, jsonify, Response
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import os
from pathlib import Path
import pandas as pd
import json
import logging
from pipeline import main as run_pipeline, OUTPUT_ROOT, MondayConfig
from config import Config

# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s ─ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# In-memory cache for latest data
latest_data = {
    "combined_csv": None,  # Path to latest combined CSV
    "group_csvs": {},      # Dict of group name to CSV path
    "json_data": None,     # Path to latest JSON
    "timestamp": None      # Timestamp of last run
}

def run_scheduled_pipeline():
    """Run the pipeline and update the in-memory cache."""
    try:
        logger.info("Starting scheduled pipeline run")
        run_pipeline()  # Run the pipeline
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = OUTPUT_ROOT / "sessions"
        
        # Update cache with latest files
        latest_data["combined_csv"] = OUTPUT_ROOT / f"combined_output_{ts}.csv"
        latest_data["json_data"] = session_dir / f"data_{ts}.json"
        latest_data["group_csvs"] = {
            g: session_dir / f"{g}_{ts}.csv"
            for g in MondayConfig.GROUP_MAPPING.values()
        }
        latest_data["timestamp"] = ts
        logger.info("Pipeline run completed, cache updated")
    except Exception as e:
        logger.error("Scheduled pipeline failed: %s", e)

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    run_scheduled_pipeline,
    "interval",
    hours=4,
    next_run_time=datetime.now()  # Run immediately on start
)
scheduler.start()

@app.route("/api/combined_csv", methods=["GET"])
def get_combined_csv():
    """Serve the latest combined CSV file."""
    if not latest_data["combined_csv"] or not latest_data["combined_csv"].exists():
        return jsonify({"error": "No combined CSV available"}), 404
    return send_file(
        latest_data["combined_csv"],
        mimetype="text/csv",
        as_attachment=True,
        download_name=latest_data["combined_csv"].name
    )

@app.route("/api/group_csv/<group>", methods=["GET"])
def get_group_csv(group):
    """Serve the latest CSV for a specific group."""
    if group not in latest_data["group_csvs"]:
        return jsonify({"error": f"Group {group} not found"}), 404
    csv_path = latest_data["group_csvs"].get(group)
    if not csv_path or not csv_path.exists():
        return jsonify({"error": f"No CSV available for group {group}"}), 404
    return send_file(
        csv_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=csv_path.name
    )

@app.route("/api/json", methods=["GET"])
def get_json():
    """Serve the latest JSON data."""
    if not latest_data["json_data"] or not latest_data["json_data"].exists():
        return jsonify({"error": "No JSON data available"}), 404
    with open(latest_data["json_data"], "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)

@app.route("/api/status", methods=["GET"])
def get_status():
    """Return the timestamp of the last pipeline run."""
    if not latest_data["timestamp"]:
        return jsonify({"status": "No data available", "timestamp": None})
    return jsonify({
        "status": "Data available",
        "timestamp": latest_data["timestamp"],
        "combined_csv": str(latest_data["combined_csv"]),
        "group_csvs": {k: str(v) for k, v in latest_data["group_csvs"].items()},
        "json_data": str(latest_data["json_data"])
    })

@app.errorhandler(Exception)
def handle_error(error):
    """Handle unexpected errors."""
    logger.error("API error: %s", error)
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
