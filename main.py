import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from src.preprocessing.pipeline import preprocessing_pipeline
# from src.postprocessing.pipeline import postprocessing_pipeline


def main():
    # postprocessing_pipeline("English", False, "UPOS", ["20", "100"])
    preprocessing_pipeline("English", ["train", "dev", "test"], [False, True], "UPOS")

if __name__ == "__main__":
    main()