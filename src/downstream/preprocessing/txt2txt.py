from tqdm import tqdm
from src.downstream.preprocessing.to_edit_actions import tree_pair_to_edit_tags, apply_edit_tags, validate_anchor_sequence

def txt2txt(src_path, tgt_path, write_path):
    with open(write_path, "w", encoding="utf-8") as fout:
        pass
    with open(src_path, "r", encoding="utf-8") as fsrc, \
         open(tgt_path, "r", encoding="utf-8") as ftgt:
         
        for src_ln, tgt_ln in tqdm(zip(fsrc, ftgt), desc="Turning into edit actions"):
            model_tokens, tags = tree_pair_to_edit_tags(src_ln, tgt_ln)
            # recovered = apply_edit_tags(model_tokens, tags)
            with open(write_path, "a", encoding="utf-8") as fout:
                fout.write(f" ".join(tags) + "\n")