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


# Required scientific-profile fields (Phase 9). Every breed exposes all of
# these via get_scientific_profile(), with missing values derived or defaulted.
SCIENTIFIC_PROFILE_FIELDS = [
    "scientific_name", "kingdom", "phylum", "class", "order", "family",
    "genus", "species", "breed", "origin_country", "native_region", "purpose",
    "average_weight_kg", "average_height_cm", "average_milk_yield_lpy",
    "temperament", "climate_adaptation", "color_pattern", "lifespan_years",
]


def _range_average(range_dict: Optional[Dict[str, Any]]) -> Optional[float]:
    """Mean of the midpoints of the male/female ranges in a {min,max} dict."""
    if not isinstance(range_dict, dict):
        return None
    midpoints = []
    for key in ("female", "male"):
        rng = range_dict.get(key)
        if isinstance(rng, (list, tuple)) and len(rng) == 2 and rng[1] > 0:
            midpoints.append((rng[0] + rng[1]) / 2.0)
    return round(sum(midpoints) / len(midpoints), 1) if midpoints else None


def get_scientific_profile(breed_name: str) -> Optional[Dict[str, Any]]:
    """
    Return a normalized scientific profile for a breed with the full field set
    required by the report/inference pipeline (Phase 9).

    Every key in :data:`SCIENTIFIC_PROFILE_FIELDS` is present; unavailable
    values are derived (genus/species from the binomial name, averages from
    ranges) or set to a sensible default rather than omitted.

    Args:
        breed_name: Breed name or a known common name/alias.

    Returns:
        Profile dict, or None if the breed is unknown.
    """
    info = get_breed_info(breed_name)
    if info is None:
        return None

    sci = info.get("scientific_name", "").strip()
    parts = sci.split()
    genus = info.get("genus") or (parts[0] if parts else "Unknown")
    species = info.get("species") or (parts[1] if len(parts) > 1 else "sp.")

    return {
        "scientific_name": sci or "Unknown",
        "kingdom": info.get("kingdom", "Animalia"),
        "phylum": info.get("phylum", "Chordata"),
        "class": info.get("class", "Mammalia"),
        "order": info.get("order", "Artiodactyla"),
        "family": info.get("family", "Bovidae"),
        "genus": genus,
        "species": species,
        "breed": info.get("breed_name", breed_name),
        "origin_country": info.get("origin_country", "Unknown"),
        "native_region": info.get("origin_region", "Unknown"),
        "purpose": info.get("purpose", "Unknown"),
        "average_weight_kg": _range_average(info.get("weight_range_kg")),
        "average_height_cm": _range_average(info.get("height_range_cm")),
        "average_milk_yield_lpy": info.get("avg_milk_yield_liters_per_year"),
        "temperament": info.get("temperament", "Unknown"),
        "climate_adaptation": info.get("climate_adaptability", "Unknown"),
        "color_pattern": info.get("coat_colors", []),
        "lifespan_years": info.get("lifespan_years", []),
        "animal_type": info.get("animal_type", "cattle"),
        "description": info.get("description", ""),
    }


def search_breeds(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Case-insensitive substring search over breed names and their common names.

    Args:
        query: Search text.
        limit: Maximum number of results.

    Returns:
        List of breed-info dicts (each with ``breed_name`` / ``animal_type``).
    """
    db = _load_database()
    query = query.strip().lower()
    if not query:
        return []
    results: List[Dict[str, Any]] = []
    for animal_type, group in (("cattle", "cattle_breeds"), ("buffalo", "buffalo_breeds")):
        for name, data in db.get(group, {}).items():
            haystack = [name.lower()] + [c.lower() for c in data.get("common_names", [])]
            if any(query in h for h in haystack):
                results.append({**data, "breed_name": name, "animal_type": animal_type})
                if len(results) >= limit:
                    return results
    return results


def get_full_taxonomy(species: str) -> Dict[str, str]:
    """
    Return the complete 8-rank taxonomy for a species, always populated.

    Falls back to the standard Bovidae/Chordata lineage when the species is
    unknown, so downstream consumers never receive missing ranks.
    """
    taxonomy = get_species_taxonomy(species) or {}
    sci = taxonomy.get("scientific_name", "").strip()
    parts = sci.split()
    return {
        "kingdom": taxonomy.get("kingdom", "Animalia"),
        "phylum": taxonomy.get("phylum", "Chordata"),
        "class": taxonomy.get("class", "Mammalia"),
        "order": taxonomy.get("order", "Artiodactyla"),
        "family": taxonomy.get("family", "Bovidae"),
        "genus": taxonomy.get("genus") or (parts[0] if parts else "Unknown"),
        "species": taxonomy.get("species") or (parts[1] if len(parts) > 1 else "sp."),
        "scientific_name": sci or "Unknown",
    }


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
