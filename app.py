"""
Flask Web Server for the AI Cattle Analysis System.

Serves the web dashboard and provides API endpoints for
image upload, analysis, and report generation.
"""

import os
import uuid
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from inference import CattleAnalysisPipeline

app = Flask(__name__)

# Initialize pipeline once to avoid reloading models on every request
print("\n" + "=" * 60)
print("  Starting AI Cattle Analysis Web Server")
print("=" * 60 + "\n")

pipeline = CattleAnalysisPipeline()

UPLOAD_FOLDER = Path("outputs/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Upload an image and run the full analysis pipeline.
    Returns comprehensive JSON with all predictions.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = app.config["UPLOAD_FOLDER"] / unique_filename
        file.save(str(file_path))

        try:
            # Run full analysis pipeline
            result_dict, annotated_image_path = pipeline.analyze(str(file_path))

            if result_dict and "error" in result_dict:
                return jsonify({"error": result_dict["error"]}), 400

            # Return comprehensive response
            return jsonify({
                "success": True,
                "results": result_dict,
                "annotated_image": Path(annotated_image_path).name,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type. Accepted: PNG, JPG, JPEG, WEBP"}), 400


@app.route("/outputs/<filename>")
def get_output_image(filename):
    """Serve annotated output images."""
    return send_from_directory(Path("outputs/inference"), filename)


@app.route("/api/breeds")
def list_breeds():
    """API endpoint to list all known breeds."""
    from database import get_all_cattle_breeds, get_all_buffalo_breeds
    cattle = list(get_all_cattle_breeds().keys())
    buffalo = list(get_all_buffalo_breeds().keys())
    return jsonify({"cattle_breeds": cattle, "buffalo_breeds": buffalo})


@app.route("/api/breed/<breed_name>")
def breed_detail(breed_name):
    """API endpoint for breed details."""
    from database import format_breed_report
    info = format_breed_report(breed_name)
    if info:
        return jsonify(info)
    return jsonify({"error": f"Breed '{breed_name}' not found"}), 404


@app.route("/api/health")
def health_check():
    """System health check."""
    return jsonify({
        "status": "healthy",
        "models": {
            "detector": "loaded",
            "species_classifier": "loaded" if pipeline.has_species_clf else "not_available",
            "breed_classifier": "loaded" if pipeline.has_breed_clf else "not_available",
            "segmentor_yolo": "loaded" if pipeline.has_yolo_seg else "not_available",
            "segmentor_unet": "loaded" if pipeline.has_unet else "not_available",
            "weight_regressor": "loaded" if pipeline.has_weight_reg else "not_available",
            "bcs_regressor": "loaded" if pipeline.has_bcs else "not_available",
        },
        "device": str(pipeline.device),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
