"""
Flask Web Server for the AI Cattle Analysis System.

Serves the web dashboard and provides API endpoints for
image upload, analysis, and report generation.
"""

import os
import math
import uuid
import json
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask.json.provider import DefaultJSONProvider
from werkzeug.utils import secure_filename
from inference import CattleAnalysisPipeline, to_json_safe


class NumpyJSONProvider(DefaultJSONProvider):
    """
    Flask JSON provider that serializes NumPy scalar/array types.

    The analysis pipeline leaves NumPy values (notably ``np.int64`` from
    OpenCV/detection ops, which — unlike ``np.float64`` — is not a Python int
    and is not JSON-serializable) throughout the result. Handling them here
    makes every ``jsonify`` response numpy-safe, regardless of source.
    """

    @staticmethod
    def default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            value = float(obj)
            return value if math.isfinite(value) else None
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return DefaultJSONProvider.default(obj)


app = Flask(__name__)
app.json = NumpyJSONProvider(app)

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

            # Render downloadable reports (JSON + PDF) when available.
            report_files = {}
            if getattr(pipeline, "report_generator", None) is not None:
                try:
                    stem = Path(unique_filename).stem
                    rendered = pipeline.report_generator.generate(
                        result_dict, formats=("json", "pdf"),
                        detection_image=annotated_image_path, basename=stem)
                    report_files = {
                        fmt: Path(path).name
                        for fmt, path in rendered["paths"].items() if path
                    }
                except Exception:  # noqa: BLE001 — reporting is best-effort
                    report_files = {}

            # Return comprehensive response. to_json_safe guards against any
            # residual NumPy scalar/array types that Flask cannot serialize.
            return jsonify(to_json_safe({
                "success": True,
                "results": result_dict,
                "annotated_image": Path(annotated_image_path).name,
                "report_files": report_files,
            }))

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type. Accepted: PNG, JPG, JPEG, WEBP"}), 400


@app.route("/outputs/<filename>")
def get_output_image(filename):
    """Serve annotated output images."""
    return send_from_directory(Path("outputs/inference"), filename)


@app.route("/reports/<filename>")
def get_report(filename):
    """Serve generated report files (JSON / PDF) as downloads."""
    return send_from_directory(Path("outputs/reports"), secure_filename(filename),
                               as_attachment=True)


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
