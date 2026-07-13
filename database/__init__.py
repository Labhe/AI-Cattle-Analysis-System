"""
Database module for the AI Cattle Analysis System.
Provides scientifically verified breed information, species taxonomy,
and lookup functions for the inference pipeline.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

DATABASE_DIR = Path(__file__).parent
BREED_DB_PATH = DATABASE_DIR / "breed_database.json"

# Cache loaded database
_db_cache: Optional[Dict] = None


def _load_database() -> Dict:
    """Load the breed database from JSON, with caching."""
    global _db_cache
    if _db_cache is None:
        with open(BREED_DB_PATH, "r", encoding="utf-8") as f:
            _db_cache = json.load(f)
    return _db_cache


def get_all_cattle_breeds() -> Dict[str, Dict]:
    """Return all cattle breed entries."""
    db = _load_database()
    return db.get("cattle_breeds", {})


def get_all_buffalo_breeds() -> Dict[str, Dict]:
    """Return all buffalo breed entries."""
    db = _load_database()
    return db.get("buffalo_breeds", {})


def get_breed_info(breed_name: str) -> Optional[Dict]:
    """
    Look up detailed information for a specific breed.
    
    Args:
        breed_name: Name of the breed (e.g., "Holstein Friesian", "Angus")
    
    Returns:
        Dict with breed details or None if not found.
    """
    db = _load_database()
    
    # Search cattle breeds
    cattle = db.get("cattle_breeds", {})
    if breed_name in cattle:
        info = cattle[breed_name].copy()
        info["breed_name"] = breed_name
        info["animal_type"] = "cattle"
        return info
    
    # Search buffalo breeds
    buffalo = db.get("buffalo_breeds", {})
    if breed_name in buffalo:
        info = buffalo[breed_name].copy()
        info["breed_name"] = breed_name
        info["animal_type"] = "buffalo"
        return info
    
    # Fuzzy match: check common_names
    for breed, data in cattle.items():
        if breed_name.lower() in [n.lower() for n in data.get("common_names", [])]:
            info = data.copy()
            info["breed_name"] = breed
            info["animal_type"] = "cattle"
            return info
    
    return None


def get_breed_names_list() -> List[str]:
    """Return a flat list of all breed names (cattle + buffalo)."""
    db = _load_database()
    names = list(db.get("cattle_breeds", {}).keys())
    names += list(db.get("buffalo_breeds", {}).keys())
    return names


def get_species_taxonomy(species: str) -> Optional[Dict]:
    """
    Get full taxonomy for a species.
    
    Args:
        species: Species name (e.g., "Cow", "Buffalo", "Horse")
    
    Returns:
        Dict with taxonomy details or None if not found.
    """
    db = _load_database()
    taxonomy = db.get("species_taxonomy", {})
    
    # Direct match
    if species in taxonomy:
        return taxonomy[species]
    
    # Case-insensitive match
    for key, val in taxonomy.items():
        if key.lower() == species.lower():
            return val
        # Check common names
        if species.lower() in [n.lower() for n in val.get("common_names", [])]:
            return val
    
    return None


def get_breeds_for_species(species: str) -> List[Dict]:
    """
    Get all known breeds for a given species.
    
    Args:
        species: Species name (e.g., "Cow", "Buffalo")
    
    Returns:
        List of breed info dicts.
    """
    db = _load_database()
    
    if species.lower() in ["cow", "cattle", "bull", "bovine", "ox", "steer"]:
        breeds = db.get("cattle_breeds", {})
        return [
            {**data, "breed_name": name}
            for name, data in breeds.items()
        ]
    elif species.lower() in ["buffalo", "water buffalo"]:
        breeds = db.get("buffalo_breeds", {})
        return [
            {**data, "breed_name": name}
            for name, data in breeds.items()
        ]
    
    return []


def get_weight_range_for_breed(breed_name: str) -> Optional[Dict]:
    """
    Get weight range for a specific breed.
    
    Returns:
        Dict with 'male' and 'female' weight ranges in kg, or None.
    """
    info = get_breed_info(breed_name)
    if info:
        return info.get("weight_range_kg")
    return None


def get_species_average_weight(species: str) -> float:
    """
    Get average weight for a species (used as fallback).
    Based on FAO/USDA published averages.
    """
    averages = {
        "cow": 550.0,
        "cattle": 550.0,
        "bull": 750.0,
        "buffalo": 500.0,
        "yak": 350.0,
        "ox": 700.0,
        "goat": 55.0,
        "sheep": 75.0,
        "horse": 500.0,
        "camel": 600.0,
        "pig": 150.0,
        "dog": 25.0,
        "cat": 4.5,
        "human": 70.0,
    }
    return averages.get(species.lower(), 100.0)


def format_breed_report(breed_name: str) -> Optional[Dict[str, Any]]:
    """
    Format a complete breed report suitable for the frontend display.
    
    Args:
        breed_name: Name of the breed
    
    Returns:
        Formatted dict with all displayable breed information.
    """
    info = get_breed_info(breed_name)
    if info is None:
        return None
    
    weight = info.get("weight_range_kg", {})
    male_weight = weight.get("male", [0, 0])
    female_weight = weight.get("female", [0, 0])
    
    height = info.get("height_range_cm", {})
    male_height = height.get("male", [0, 0])
    female_height = height.get("female", [0, 0])
    
    lifespan = info.get("lifespan_years", [0, 0])
    milk_yield = info.get("avg_milk_yield_liters_per_year")
    
    return {
        "breed_name": info.get("breed_name", breed_name),
        "scientific_name": info.get("scientific_name", "Unknown"),
        "family": info.get("family", "Unknown"),
        "order": info.get("order", "Unknown"),
        "class": info.get("class", "Mammalia"),
        "kingdom": info.get("kingdom", "Animalia"),
        "origin_country": info.get("origin_country", "Unknown"),
        "origin_region": info.get("origin_region", "Unknown"),
        "purpose": info.get("purpose", "Unknown"),
        "description": info.get("description", ""),
        "weight_range": f"{male_weight[0]}-{male_weight[1]} kg (male), {female_weight[0]}-{female_weight[1]} kg (female)",
        "height_range": f"{male_height[0]}-{male_height[1]} cm (male), {female_height[0]}-{female_height[1]} cm (female)" if male_height[0] > 0 else "Unknown",
        "avg_milk_yield": f"{milk_yield} L/year" if milk_yield else "N/A (Beef breed)",
        "coat_colors": info.get("coat_colors", []),
        "temperament": info.get("temperament", "Unknown"),
        "climate_adaptability": info.get("climate_adaptability", "Unknown"),
        "lifespan": f"{lifespan[0]}-{lifespan[1]} years" if lifespan[0] > 0 else "Unknown",
        "horn_status": info.get("horn_status", "Unknown"),
    }
