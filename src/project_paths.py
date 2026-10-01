from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"

CRSS_YEARS = tuple(range(2020, 2025))
CRSS_TRAIN_YEARS = (2020, 2021, 2022)
CRSS_VALIDATION_YEAR = 2023
CRSS_TEST_YEAR = 2024


def crss_extracted_dir(year: int) -> Path:
    """Return the directory containing the 28 CSV files for one CRSS year."""
    year_root = RAW_DIR / "crss" / str(year) / f"CRSS{year}CSV"
    candidates = [year_root / f"CRSS{year}CSV", year_root]
    for candidate in candidates:
        if (candidate / "accident.csv").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate extracted CRSS files for {year}: {year_root}")


def complaint_file() -> Path:
    path = RAW_DIR / "complaints" / "FLAT_CMPL" / "FLAT_CMPL.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


for directory in (INTERIM_DIR, PROCESSED_DIR, TABLE_DIR, FIGURE_DIR, EVALUATION_DIR):
    directory.mkdir(parents=True, exist_ok=True)
