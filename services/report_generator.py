"""
Professional Report Generator for the AI Cattle Analysis System (Phase 10).

Turns an inference result (the dict produced by
``inference.CattleAnalysisPipeline.analyze``) into a structured, professional
livestock report and renders it to JSON and/or PDF.

The report consolidates:
  - Detected species, breed, and scientific name
  - Estimated weight (with interval) and Body Condition Score (with uncertainty)
  - Prediction confidences
  - Full 8-rank taxonomy and the breed's scientific profile
  - Physical (morphometric) body measurements
  - Health indicators derived from BCS and image quality
  - The annotated detection image and segmentation overlay
  - A prediction timestamp

PDF output uses matplotlib (already a project dependency) so it works without
any extra packages; if ``reportlab`` is installed it is used for a richer
layout. JSON output is always available.

Usage:
    from services.report_generator import ReportGenerator

    generator = ReportGenerator()
    outputs = generator.generate(analysis_result, formats=("json", "pdf"),
                                 detection_image=annotated_path,
                                 segmentation_overlay=overlay_path)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "model_config.yaml"

logger = logging.getLogger("services.report_generator")

# BCS → health-status bands (standard 1–5 body condition interpretation).
BCS_HEALTH_BANDS: List[Tuple[float, float, str, str]] = [
    (1.0, 2.0, "Underweight", "Emaciated / thin — nutritional intervention advised"),
    (2.0, 2.5, "Below Ideal", "Slightly thin — monitor feeding"),
    (2.5, 3.5, "Ideal", "Healthy body condition"),
    (3.5, 4.0, "Above Ideal", "Slightly over-conditioned — monitor intake"),
    (4.0, 5.0, "Overweight", "Over-conditioned / obese — dietary management advised"),
]


def bcs_health_status(bcs: Optional[float]) -> Dict[str, str]:
    """Map a BCS value to a health category and recommendation."""
    if bcs is None:
        return {"category": "Unknown", "recommendation": "BCS unavailable"}
    for lo, hi, category, recommendation in BCS_HEALTH_BANDS:
        if lo <= bcs < hi or (hi == 5.0 and bcs == 5.0):
            return {"category": category, "recommendation": recommendation}
    return {"category": "Unknown", "recommendation": "BCS out of range"}


class ReportGenerator:
    """
    Build and render professional livestock analysis reports.

    Args:
        config: Parsed ``model_config.yaml`` dict; loaded from disk if omitted.
        output_dir: Where reports are written; defaults to ``outputs/reports``.
        organization: Title shown in the report header.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
        organization: str = "AI Cattle Analysis System",
    ):
        self.config = config if config is not None else self._load_config()
        report_cfg = self.config.get("report", {})
        self.output_dir = Path(output_dir) if output_dir \
            else BASE_DIR / report_cfg.get("output_dir", "outputs/reports")
        self.organization = report_cfg.get("organization", organization)

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        if DEFAULT_CONFIG_PATH.exists():
            with open(DEFAULT_CONFIG_PATH, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ─────────────────────────────── build ───────────────────────────────

    def build_report(
        self,
        analysis: Dict[str, Any],
        detection_image: Optional[str] = None,
        segmentation_overlay: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Normalize an inference result into the structured report schema.

        Args:
            analysis: Result dict from the inference pipeline.
            detection_image: Path to the annotated detection image.
            segmentation_overlay: Path to the segmentation overlay image.
            timestamp: ISO timestamp; defaults to now (UTC).

        Returns:
            A structured, JSON-serializable report dict.
        """
        analysis = analysis or {}
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        bcs = analysis.get("bcs")
        health = bcs_health_status(bcs)
        quality = analysis.get("image_quality", {}) or {}

        profile = analysis.get("scientific_profile") or {}
        taxonomy = analysis.get("taxonomy", {}) or {}

        return {
            "report_metadata": {
                "organization": self.organization,
                "generated_at": ts,
                "image": analysis.get("image", "unknown"),
                "schema_version": "1.0",
            },
            "species": {
                "name": analysis.get("species", "Unknown"),
                "confidence_pct": analysis.get("species_confidence", 0.0),
                "top_5": analysis.get("species_top_5", analysis.get("species_top_3", [])),
            },
            "breed": {
                "name": analysis.get("breed", "Unknown"),
                "confidence_pct": analysis.get("breed_confidence", 0.0),
                "scientific_name": profile.get("scientific_name",
                                               taxonomy.get("scientific_name", "Unknown")),
                "top_5": analysis.get("breed_top_5", []),
                "note": analysis.get("breed_message"),
            },
            "taxonomy": {
                "kingdom": taxonomy.get("kingdom", profile.get("kingdom", "Animalia")),
                "phylum": taxonomy.get("phylum", profile.get("phylum", "Chordata")),
                "class": taxonomy.get("class", profile.get("class", "Mammalia")),
                "order": taxonomy.get("order", profile.get("order", "Artiodactyla")),
                "family": taxonomy.get("family", profile.get("family", "Bovidae")),
                "genus": profile.get("genus", "Unknown"),
                "species": profile.get("species", "Unknown"),
            },
            "weight": {
                "estimated_kg": analysis.get("weight_kg"),
                "confidence_pct": analysis.get("weight_confidence", 0.0),
                "interval_kg": analysis.get("weight_interval_kg"),
                "method": analysis.get("weight_method", "unknown"),
            },
            "body_condition_score": {
                "score": bcs,
                "confidence_pct": analysis.get("bcs_confidence", 0.0),
                "uncertainty": analysis.get("bcs_uncertainty"),
                "method": analysis.get("bcs_method", "unknown"),
            },
            "physical_measurements": analysis.get("measurements", {}),
            "health_indicators": {
                "bcs_category": health["category"],
                "recommendation": health["recommendation"],
                "image_quality": quality.get("quality", quality.get("status", "unknown")),
                "quality_issues": quality.get("issues", []),
            },
            "scientific_profile": profile,
            "images": {
                "detection": detection_image,
                "segmentation_overlay": segmentation_overlay,
            },
            "detection": {
                "box": analysis.get("detection", {}).get("box"),
                "confidence_pct": analysis.get("detection", {}).get("confidence"),
            },
        }

    # ─────────────────────────────── render ───────────────────────────────

    def to_json(self, report: Dict[str, Any], path: Path) -> Path:
        """Write the report as pretty-printed JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return path

    def to_pdf(self, report: Dict[str, Any], path: Path) -> Optional[Path]:
        """
        Render the report as a PDF.

        Uses ``reportlab`` when available, otherwise matplotlib (always present).
        Returns the path, or None if PDF rendering is unavailable.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return self._pdf_reportlab(report, path)
        except ImportError:
            pass
        try:
            return self._pdf_matplotlib(report, path)
        except Exception as e:  # noqa: BLE001 — PDF is best-effort
            logger.error(f"PDF generation failed: {e}")
            return None

    def _pdf_reportlab(self, report: Dict[str, Any], path: Path) -> Path:
        """Rich PDF layout using reportlab (raises ImportError if absent)."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(path), pagesize=A4)
        width, height = A4
        y = height - 2 * cm

        c.setFont("Helvetica-Bold", 18)
        c.drawString(2 * cm, y, report["report_metadata"]["organization"])
        y -= 0.8 * cm
        c.setFont("Helvetica", 10)
        c.drawString(2 * cm, y, f"Generated: {report['report_metadata']['generated_at']}")
        y -= 1.0 * cm

        for title, lines in self._report_sections(report):
            if y < 3 * cm:
                c.showPage()
                y = height - 2 * cm
            c.setFont("Helvetica-Bold", 12)
            c.drawString(2 * cm, y, title)
            y -= 0.6 * cm
            c.setFont("Helvetica", 10)
            for line in lines:
                c.drawString(2.3 * cm, y, line)
                y -= 0.5 * cm
            y -= 0.3 * cm
        c.save()
        return path

    def _pdf_matplotlib(self, report: Dict[str, Any], path: Path) -> Path:
        """Fallback PDF using matplotlib (text page + embedded images)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(str(path)) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            fig.text(0.07, 0.95, report["report_metadata"]["organization"],
                     fontsize=18, fontweight="bold")
            fig.text(0.07, 0.925, f"Generated: {report['report_metadata']['generated_at']}",
                     fontsize=9, color="gray")

            y = 0.89
            for title, lines in self._report_sections(report):
                fig.text(0.07, y, title, fontsize=12, fontweight="bold", color="#1a5276")
                y -= 0.028
                for line in lines:
                    fig.text(0.10, y, line, fontsize=9)
                    y -= 0.022
                y -= 0.012
                if y < 0.08:
                    break
            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

            # Second page: detection image + segmentation overlay.
            images = report.get("images", {})
            valid = [(name, p) for name, p in images.items() if p and Path(p).exists()]
            if valid:
                fig2, axes = plt.subplots(len(valid), 1, figsize=(8.27, 11.69))
                if len(valid) == 1:
                    axes = [axes]
                for ax, (name, p) in zip(axes, valid):
                    try:
                        ax.imshow(mpimg.imread(p))
                    except Exception:  # noqa: BLE001
                        ax.text(0.5, 0.5, f"[{name} unavailable]", ha="center")
                    ax.set_title(name.replace("_", " ").title())
                    ax.axis("off")
                pdf.savefig(fig2)
                plt.close(fig2)
        return path

    def _report_sections(self, report: Dict[str, Any]) -> Iterable[Tuple[str, List[str]]]:
        """Yield (section title, text lines) pairs for PDF rendering."""
        sp = report["species"]
        yield "Species", [
            f"Detected: {sp['name']}  ({sp['confidence_pct']}% confidence)",
        ]
        br = report["breed"]
        breed_lines = [f"Breed: {br['name']}  ({br['confidence_pct']}% confidence)",
                       f"Scientific name: {br['scientific_name']}"]
        if br.get("note"):
            breed_lines.append(str(br["note"]))
        yield "Breed", breed_lines

        tx = report["taxonomy"]
        yield "Taxonomy", [
            f"{rank.title()}: {tx.get(rank, 'Unknown')}"
            for rank in ("kingdom", "phylum", "class", "order", "family", "genus", "species")
        ]
        w = report["weight"]
        interval = w.get("interval_kg")
        interval_str = f"  interval {interval[0]}–{interval[1]} kg" if interval else ""
        yield "Weight Estimation", [
            f"Estimated weight: {w['estimated_kg']} kg  ({w['confidence_pct']}% confidence){interval_str}",
            f"Method: {w['method']}",
        ]
        b = report["body_condition_score"]
        yield "Body Condition Score", [
            f"BCS: {b['score']}  ({b['confidence_pct']}% confidence)",
            f"Uncertainty: {b.get('uncertainty', 'N/A')}   Method: {b['method']}",
        ]
        yield "Physical Measurements", [
            f"{k.replace('_', ' ').title()}: {v}"
            for k, v in report["physical_measurements"].items()
        ] or ["No measurements available"]
        h = report["health_indicators"]
        yield "Health Indicators", [
            f"BCS category: {h['bcs_category']}",
            f"Recommendation: {h['recommendation']}",
            f"Image quality: {h['image_quality']}",
        ]

    # ─────────────────────────────── generate ───────────────────────────────

    def generate(
        self,
        analysis: Dict[str, Any],
        formats: Iterable[str] = ("json", "pdf"),
        detection_image: Optional[str] = None,
        segmentation_overlay: Optional[str] = None,
        basename: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build and render the report in the requested formats.

        Args:
            analysis: Inference result dict.
            formats: Any of ``"json"``, ``"pdf"``.
            detection_image: Path to the annotated detection image.
            segmentation_overlay: Path to the segmentation overlay image.
            basename: Output filename stem; defaults to the image stem + time.
            timestamp: ISO timestamp override.

        Returns:
            Dict with the structured ``report`` and written file ``paths``.
        """
        report = self.build_report(analysis, detection_image, segmentation_overlay, timestamp)
        stem = basename or (Path(analysis.get("image", "report")).stem or "report")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Optional[str]] = {}
        formats = {f.lower() for f in formats}
        if "json" in formats:
            paths["json"] = str(self.to_json(report, self.output_dir / f"{stem}_report.json"))
        if "pdf" in formats:
            pdf_path = self.to_pdf(report, self.output_dir / f"{stem}_report.pdf")
            paths["pdf"] = str(pdf_path) if pdf_path else None
        return {"report": report, "paths": paths}
