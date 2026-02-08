from src.downstream.preprocessing.linearize import linearize
from tqdm import tqdm

def mrg2txt(read_path, write_path):
    with open(write_path, "w", encoding="utf-8") as fout:
        pass

    with open(read_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Linearizing"):
            linearzied = linearize(line)
            with open(write_path, "a", encoding="utf=8") as fout:
                fout.write(linearzied)
                fout.write("\n")