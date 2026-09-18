import json
import re
from tqdm import tqdm
from src.data.downstream.preprocessing.to_edit_actions import tree_pair_to_edit_tags
from src.data.downstream.preprocessing.to_structured_tokens import linearized_tree_to_structured_tokens

def txt_to_edit_actions_json(src_path, tgt_path, write_path):

    examples = []
    with open(src_path, "r", encoding="utf-8") as fsrc, \
         open(tgt_path, "r", encoding="utf-8") as ftgt:
        
        for src_ln, tgt_ln in tqdm(zip(fsrc, ftgt), desc="Turning into edit actions"):
            examples.append(tree_pair_to_edit_tags(src_ln, tgt_ln))   
            
    with open(write_path, "w", encoding="utf-8") as fout:
        json.dump(examples, fout, indent=4)

def txt_to_structured_tokens_json(read_src_path, read_tgt_path, write_path, overlap=1, read_lang_path=None):
     
    # src_examples = []
    # tgt_examples = []
    examples = []

    if read_lang_path is not None:

        with open(read_src_path, "r", encoding="utf-8") as fsrcin, \
             open(read_tgt_path, "r", encoding="utf-8") as ftgtin, \
             open(read_lang_path, "r", encoding="utf-8") as flangin:
            
            # print(len(list(fsrcin)))
            # print(len(list(ftgtin)))
            # print(len(list(flangin)))
            for src_ln, tgt_ln, lang_ln in tqdm(zip(fsrcin, ftgtin, flangin, strict=True), desc="Turning into structured tokens"):
                examples.append({
                    "lang": lang_ln.strip(),
                    "source":{
                        "base": linearized_tree_to_structured_tokens(src_ln, overlap=0),
                        "overlap": linearized_tree_to_structured_tokens(src_ln, overlap=overlap),
                        },
                    "target": {
                        "local": linearized_tree_to_structured_tokens(tgt_ln, overlap=0),
                        # "global": tgt_ln.split(),
                        },
                })
    else: 
        with open(read_src_path, "r", encoding="utf-8") as fsrcin, \
             open(read_tgt_path, "r", encoding="utf-8") as ftgtin:
            lang = re.search(r"(?<=lang=)(.*?)(?=,)", str(read_src_path)).group()
            
            for src_ln, tgt_ln in tqdm(zip(fsrcin, ftgtin, strict=True), desc="Turning into structured tokens"):
                examples.append({
                    "lang": lang,
                    "source":{
                        "base": linearized_tree_to_structured_tokens(src_ln, overlap=0),
                        "overlap": linearized_tree_to_structured_tokens(src_ln, overlap=overlap),
                        },
                    "target": {
                        "local": linearized_tree_to_structured_tokens(tgt_ln, overlap=0),
                        # "global": tgt_ln.split(),
                        },
                })
            
    
    with open(write_path, "w", encoding="utf-8") as fout:
        #  open(write_tgt_path, "w", encoding="utf-8") as ftgtout:
        json.dump(examples, fout, indent=2)
      
            

    