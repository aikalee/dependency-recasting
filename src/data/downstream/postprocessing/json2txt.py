import json
from tqdm import tqdm
from src.data.downstream.postprocessing.to_linearized import structured_tokens_to_linearzied


def json2txt(read_path, write_path):
    with open(write_path, "w", encoding="utf-8") as fout:
        pass
    with open(read_path, "r", encoding="utf-8") as fin:
        preds = json.load(fin)
        for ex in tqdm(preds, desc="Structured tokens to linearized"):
            linearized = structured_tokens_to_linearzied(ex)
            with open(write_path, "a", encoding="utf-8") as fout:
                fout.write(linearized + "\n")
