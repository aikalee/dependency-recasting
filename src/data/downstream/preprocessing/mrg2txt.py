from src.data.downstream.preprocessing.linearize import linearize
from tqdm import tqdm

def mrg2txt(read_path, write_path, add_bos=False):
    with open(write_path, "w", encoding="utf-8") as fout:
        pass

    with open(read_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Linearizing"):
            linearzied = linearize(line, add_bos=add_bos)
            with open(write_path, "a", encoding="utf=8") as fout:
                fout.write(linearzied)
                fout.write("\n")

