from itertools import product
from tqdm import tqdm
from src.pathgen import get_linearized_txt_path,  get_combined_txt_path

import json
import numpy as np
import random
import re

random.seed(42)
np.random.seed(42)

LANGUAGES = [ "Ancient_Greek", "English-EWT", "English-Penn", "Finnish", "French", "Hebrew", "Russian",  "Tamil",  "Uyghur", "Wolof"]
SPLITS = ["train", "dev"]

def get_sentence_ids(src_examples, tgt_examples):
    illformed = []
    wellformed = []
    for i, (src_ln, tgt_ln) in enumerate(zip(src_examples, tgt_examples)):
        if src_ln.strip() == tgt_ln.strip():
            wellformed.append(i)
        else:
            illformed.append(i)
    return illformed, wellformed

def combine_train_sentences(lang, src_path, tgt_path, src_write_path, tgt_write_path):
    
    with open(src_path, "r", encoding="utf-8") as fsrc, \
         open(tgt_path, "r", encoding="utf-8") as ftgt:
        # src_lines = fsrc.readlines()
        # tgt_lines = ftgt.readlines()
        src_examples = json.load(fsrc)
        tgt_examples = json.load(ftgt)
    # add language here
    
    illformed = [(src_ex, tgt_ex) for src_ex, tgt_ex in zip(src_examples, tgt_examples) if src_ex["global"].strip() != tgt_ln.strip()]
    wellformed = [({lang: src_ln}, {lang: tgt_ln}) for src_ln, tgt_ln in zip(src_examples, tgt_examples) if src_ln.strip() == tgt_ln.strip()]
    if len(illformed) < 2 or len(wellformed) < 1:
        raise ValueError("Sample size too small.")
    idx = np.random.randint(len(wellformed), size=len(illformed))
    wellformed = np.array(wellformed)
    selected_wellformed = wellformed[idx]
    final_list = illformed + selected_wellformed.tolist()
    random.shuffle(final_list)
    with open(src_write_path, "a", encoding="utf-8") as fsrcout, \
         open(tgt_write_path, "a", encoding="utf-8") as ftgtout:
            json.write(list(zip(*final_list))[0], fsrcout, indent=4)
            json.write(list(zip(*final_list))[1], ftgtout, indent=4)
            # fsrcout.write("".join(list(zip(*final_list))[0]))
            # ftgtout.write("".join(list(zip(*final_list))[1]))

    return len(final_list)

def combine_dev_sentences(lang, src_path, tgt_path, src_write_path, tgt_write_path, dev_count):
    
    with open(src_path, "r", encoding="utf-8") as fsrc, \
         open(tgt_path, "r", encoding="utf-8") as ftgt:
        src_lines = fsrc.readlines()
        tgt_lines = ftgt.readlines()
    dev_count = 2 if dev_count < 2 else dev_count
    illformed = np.array([({lang: src_ln}, {lang: tgt_ln}) for src_ln, tgt_ln in zip(src_lines, tgt_lines) if src_ln.strip() != tgt_ln.strip()])
    illformed_idx = np.random.randint(len(illformed), size=round(dev_count/2))
    selected_illformed = illformed[illformed_idx]
    wellformed = np.array([({lang: src_ln}, {lang: tgt_ln}) for src_ln, tgt_ln in zip(src_lines, tgt_lines) if src_ln.strip() == tgt_ln.strip()])
    wellformed_idx = np.random.randint(len(wellformed), size=round(dev_count/2))
    selected_wellformed = wellformed[wellformed_idx]
    final_list = selected_illformed.tolist() + selected_wellformed.tolist()
    random.shuffle(final_list)
    with open(src_write_path, "a", encoding="utf-8") as fsrcout, \
         open(tgt_write_path, "a", encoding="utf-8") as ftgtout:
            fsrcout.write("".join(list(zip(*final_list))[0]))
            ftgtout.write("".join(list(zip(*final_list))[1]))
    return len(final_list)



    

    # with open(src_path, "r", encoding="utf-8") as fsrcin, \
    #      open(tgt_path, "r", encoding="utf-8") as ftgtin:
    #     for src_ln, tgt_ln in tqdm(zip(fsrcin, ftgtin, strict=True), desc="Combining datasets"):
    #         if src_ln.strip() != tgt_ln.strip():
    #             illformed += 1
    #             total += 1
    #             with open(src_write_path, "a", encoding="utf-8") as fsrcout, \
    #                  open(tgt_write_path, "a", encoding="utf-8") as ftgtout:
    #                     fsrcout.write(src_ln + "\n")
    #                     ftgtout.write(tgt_ln + "\n")
    #         elif illformed > wellformed and wellformed/total != 1/3:
    #             wellformed += 1
    #             total += 1
    #             with open(src_write_path, "a", encoding="utf-8") as fsrcout, \
    #                  open(tgt_write_path, "a", encoding="utf-8") as ftgtout:
    #                     fsrcout.write(src_ln + "\n")
    #                     ftgtout.write(tgt_ln + "\n")
    #         # else:
    #         #     continue
    # print(illformed, wellformed)
    # if wellformed != 0:
    #     percent = (illformed/total)*100
    #     print(f"Percentage: {percent:.3f}")
    # return illformed+wellformed

  


def combine_datasets(langs=LANGUAGES, splits=SPLITS):
    total_train_count = 0
    total_dev_count = 0
   
    langs = langs if isinstance(langs, list) else [langs]
    splits = splits if isinstance(splits, list) else [splits]
    
    # === build blank files ===
    for split in splits:

        src_write_path = get_combined_txt_path(split, is_target=False)
        tgt_write_path = get_combined_txt_path(split, is_target=True)

        with open(src_write_path, "w", encoding="utf-8") as fsrcout, \
             open(tgt_write_path, "w", encoding="utf-8") as ftgtout:
                pass
           

    for lang in langs:
        train_count = 0
        dev_count = 0

        for split in splits:
            src_path = get_linearized_txt_path(lang=lang, pos="upos", split=split, is_target=False)
            tgt_path = get_linearized_txt_path(lang=lang, pos="upos", split=split, is_target=True)
            print(f"Loading from {src_path} and {tgt_path}...")

            src_write_path = get_combined_txt_path(split, is_target=False)
            tgt_write_path = get_combined_txt_path(split, is_target=True)

           
            if split == "train":
                lang_train_count = combine_train_sentences(src_path, tgt_path, src_write_path, tgt_write_path)
                expected_dev_count = round(lang_train_count/0.8*0.2)
                train_count += lang_train_count
                total_train_count += train_count
               
            elif split == "dev":
                lang_dev_count = combine_dev_sentences(src_path, tgt_path, src_write_path, tgt_write_path, dev_count=expected_dev_count)
                dev_count += lang_dev_count
                total_dev_count += dev_count
     
            
        print(f"Split: {split}\nTotal number of sentences in train dataset: {train_count}\nTotal number in dev dataset{dev_count}")
    print(total_train_count, total_dev_count)
