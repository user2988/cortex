"""
Cortex — Cronometer Nutrition Integration
Authenticates with Cronometer, downloads the daily summary CSV for yesterday,
parses all 84 nutrients, and writes them to the PostgreSQL nutrition table.
"""

import csv
import io
import os
import requests

from db import get_conn

CRONOMETER_EMAIL    = os.environ.get("CRONOMETER_EMAIL")
CRONOMETER_PASSWORD = os.environ.get("CRONOMETER_PASSWORD")


# ─────────────────────────────────────────────────────────────
# PART 1 — CRONOMETER CLIENT
# ─────────────────────────────────────────────────────────────

class CronometerClient:
    BASE = "https://cronometer.com"

    def __init__(self, email, password):
        self.session = requests.Session()
        self._login(email, password)

    def _login(self, email, password):
        resp = self.session.post(f"{self.BASE}/api/auth", json={
            "username": email,
            "password": password,
        })
        resp.raise_for_status()

    def fetch_csv(self, d):
        """Download the daily summary CSV for date d (YYYY-MM-DD)."""
        resp = self.session.get(f"{self.BASE}/api/food/export", params={
            "start":    d,
            "end":      d,
            "generate": 3,   # 3 = daily summary
        })
        resp.raise_for_status()
        return resp.text


# ─────────────────────────────────────────────────────────────
# PART 2 — CSV PARSING
# ─────────────────────────────────────────────────────────────

# Cronometer CSV column name → PostgreSQL nutrition column
# If any columns come back NULL after the first real run, check the
# actual header row in the export and adjust names here accordingly.
COLUMN_MAP = {
    # Energy & macros
    "Energy (kcal)":                "calories_in",
    "Protein (g)":                  "protein_g",
    "Carbohydrates (g)":            "carbs_g",
    "Fat (g)":                      "fat_g",
    "Fiber (g)":                    "fibre_g",
    "Sugars (g)":                   "sugar_g",
    "Sodium (mg)":                  "sodium_mg",
    "Water (mL)":                   "water_ml",
    # Fat subtypes
    "Saturated Fat (g)":            "saturated_fat_g",
    "Monounsaturated Fat (g)":      "monounsaturated_fat_g",
    "Polyunsaturated Fat (g)":      "polyunsaturated_fat_g",
    "Trans-Fatty Acids (g)":        "trans_fat_g",
    "Cholesterol (mg)":             "cholesterol_mg",
    # Stimulants
    "Caffeine (mg)":                "caffeine_mg",
    # Omega fatty acids — Cronometer exports in grams, DB stores in mg
    "Omega-3 (g)":                  "omega3_mg",
    "Omega-6 (g)":                  "omega6_mg",
    "ALA (18:3) (g)":               "ala_mg",
    "EPA (20:5) (g)":               "epa_mg",
    "DHA (22:6) (g)":               "dha_mg",
    # Fat-soluble vitamins
    "Vitamin A (mcg RAE)":          "vitamin_a_mcg",
    "Vitamin D (IU)":               "vitamin_d_iu",
    "Vitamin E (mg)":               "vitamin_e_mg",
    "Vitamin K (mcg)":              "vitamin_k_mcg",
    # Water-soluble vitamins
    "Vitamin C (mg)":               "vitamin_c_mg",
    "B1 (Thiamine) (mg)":           "thiamine_mg",
    "B2 (Riboflavin) (mg)":         "riboflavin_mg",
    "B3 (Niacin) (mg)":             "niacin_mg",
    "B5 (Pantothenic Acid) (mg)":   "pantothenic_acid_mg",
    "B6 (mg)":                      "vitamin_b6_mg",
    "B7 (Biotin) (mcg)":            "biotin_mcg",
    "B9 (Folate) (mcg)":            "folate_mcg",
    "B12 (mcg)":                    "vitamin_b12_mcg",
    # Minerals
    "Calcium (mg)":                 "calcium_mg",
    "Iron (mg)":                    "iron_mg",
    "Magnesium (mg)":               "magnesium_mg",
    "Phosphorus (mg)":              "phosphorus_mg",
    "Potassium (mg)":               "potassium_mg",
    "Zinc (mg)":                    "zinc_mg",
    # Trace minerals
    "Selenium (mcg)":               "selenium_mcg",
    "Copper (mg)":                  "copper_mg",
    "Manganese (mg)":               "manganese_mg",
    "Chromium (mcg)":               "chromium_mcg",
    "Iodine (mcg)":                 "iodine_mcg",
    "Molybdenum (mcg)":             "molybdenum_mcg",
    # Essential amino acids
    "Tryptophan (g)":               "tryptophan_g",
    "Threonine (g)":                "threonine_g",
    "Isoleucine (g)":               "isoleucine_g",
    "Leucine (g)":                  "leucine_g",
    "Lysine (g)":                   "lysine_g",
    "Methionine (g)":               "methionine_g",
    "Phenylalanine (g)":            "phenylalanine_g",
    "Valine (g)":                   "valine_g",
    "Histidine (g)":                "histidine_g",
    # Non-essential amino acids
    "Alanine (g)":                  "alanine_g",
    "Arginine (g)":                 "arginine_g",
    "Aspartic Acid (g)":            "aspartic_acid_g",
    "Cystine (g)":                  "cystine_g",
    "Glutamic Acid (g)":            "glutamic_acid_g",
    "Glycine (g)":                  "glycine_g",
    "Proline (g)":                  "proline_g",
    "Serine (g)":                   "serine_g",
    "Tyrosine (g)":                 "tyrosine_g",
    "Hydroxyproline (g)":           "hydroxyproline_g",
}

# These are exported in grams but the DB column stores milligrams
G_TO_MG = {"omega3_mg", "omega6_mg", "ala_mg", "epa_mg", "dha_mg"}

# Alcohol: Cronometer exports grams of ethanol. 1 US standard drink = 14g.
ALCOHOL_G_PER_UNIT = 14.0


def parse_csv(csv_text, d):
    """
    Parse Cronometer daily summary CSV text for date d (YYYY-MM-DD).
    Returns a record dict ready for store_nutrition(), or None if no row found.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        if row.get("Date") != d:
            continue

        record = {"date": d}

        for crono_col, db_col in COLUMN_MAP.items():
            raw = row.get(crono_col)
            if raw in (None, ""):
                record[db_col] = None
            else:
                try:
                    val = float(raw)
                    record[db_col] = val * 1000 if db_col in G_TO_MG else val
                except ValueError:
                    record[db_col] = None

        # Alcohol: grams → standard drinks
        alc = row.get("Alcohol (g)")
        record["alcohol_units"] = round(float(alc) / ALCOHOL_G_PER_UNIT, 2) if alc else None

        # caffeine_last_time is not in the CSV export — future manual or supplement logger feature
        record["caffeine_last_time"] = None

        return record

    return None  # no row found for this date


# ─────────────────────────────────────────────────────────────
# PART 3 — POSTGRESQL STORAGE
# ─────────────────────────────────────────────────────────────

def store_nutrition(record):
    """
    Upsert a nutrition record into PostgreSQL.
    SQL is built dynamically from the record keys so the column map drives everything.
    """
    cols        = [k for k in record if k != "date"]
    col_list    = ", ".join(["date"] + cols)
    val_list    = ", ".join(["%(date)s"] + [f"%({c})s" for c in cols])
    update_list = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols)

    sql = f"""
        INSERT INTO nutrition ({col_list})
        VALUES ({val_list})
        ON CONFLICT (date) DO UPDATE SET {update_list};
    """

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, record)
        print(f"  Nutrition stored for {record['date']}.")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# PART 4 — PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_nutrition_pipeline(d):
    """
    Fetch and store Cronometer nutrition data for date d.
    Skips gracefully if credentials are not configured.
    Called from cortex.py after biometrics are stored.
    """
    if not CRONOMETER_EMAIL or not CRONOMETER_PASSWORD:
        print("  Cronometer credentials not set — skipping nutrition.")
        return

    try:
        client   = CronometerClient(CRONOMETER_EMAIL, CRONOMETER_PASSWORD)
        csv_text = client.fetch_csv(d)
        record   = parse_csv(csv_text, d)
        if record:
            store_nutrition(record)
        else:
            print(f"  Cronometer: no data found for {d}.")
    except Exception as e:
        print(f"  Cronometer FAILED: {e}")
