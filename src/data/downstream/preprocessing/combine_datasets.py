from itertools import product
from tqdm import tqdm
# from src.pathgen import get_linearized_txt_path,  get_combined_txt_path, get_combined_langs_txt_path
from src.pathgen import UpstreamPredictionPaths

import json
import numpy as np
import random
import re

random.seed(42)
np.random.seed(42)

LANGUAGES = [ "Ancient_Greek", "English-EWT", "English-Penn", "Finnish", "French", "Hebrew", "Russian",  "Tamil",  "Uyghur", "Wolof"]
SPLITS = ["train", "dev"]


def combine_train_sentences(src_path, tgt_path, src_write_path, tgt_write_path, lang_write_path):
    lang = re.search(r"(?<=lang=)(.*?)(?=,)", str(src_path)).group()
  
    with open(src_path, "r", encoding="utf-8") as fsrc, \
         open(tgt_path, "r", encoding="utf-8") as ftgt:
        src_lines = fsrc.readlines()
        tgt_lines = ftgt.readlines()
        # src_examples = json.load(fsrc)
        # tgt_examples = json.load(ftgt)
    illformed = [(src_ln, tgt_ln) for src_ln, tgt_ln in zip(src_lines, tgt_lines) if src_ln.strip() != tgt_ln.strip()]
    wellformed = np.array([(src_ln, tgt_ln) for src_ln, tgt_ln in zip(src_lines, tgt_lines) if src_ln.strip() == tgt_ln.strip()])
    if len(illformed) < 2 or len(wellformed) < 1:
        raise ValueError("Sample size too small.")
    
    # === select random wellformed sentence ===
    idx = np.random.randint(len(wellformed), size=len(illformed))
    selected_wellformed = wellformed[idx]

    # === shuffle ===
    final_list = illformed + selected_wellformed.tolist()
    random.shuffle(final_list)

    langs = [lang] * len(final_list)
    
   
    with open(src_write_path, "a", encoding="utf-8") as fsrcout, \
         open(tgt_write_path, "a", encoding="utf-8") as ftgtout, \
         open(lang_write_path, "a", encoding="utf-8") as flangout:
            fsrcout.write("".join(list(zip(*final_list))[0]))
            ftgtout.write("".join(list(zip(*final_list))[1]))
            flangout.write("\n".join(langs) + "\n")

    return len(final_list)

def combine_dev_sentences(src_path, tgt_path, src_write_path, tgt_write_path, lang_write_path, dev_count):
    lang = re.search(r"(?<=lang=)(.*?)(?=,)", str(src_path)).group()
    with open(src_path, "r", encoding="utf-8") as fsrc, \
         open(tgt_path, "r", encoding="utf-8") as ftgt:
        src_lines = fsrc.readlines()
        tgt_lines = ftgt.readlines()
    dev_count = 2 if dev_count < 2 else dev_count
    illformed = np.array([(src_ln, tgt_ln) for src_ln, tgt_ln in zip(src_lines, tgt_lines) if src_ln.strip() != tgt_ln.strip()])
    illformed_idx = np.random.randint(len(illformed), size=round(dev_count/2))
    selected_illformed = illformed[illformed_idx]
    wellformed = np.array([(src_ln, tgt_ln) for src_ln, tgt_ln in zip(src_lines, tgt_lines) if src_ln.strip() == tgt_ln.strip()])
    wellformed_idx = np.random.randint(len(wellformed), size=round(dev_count/2))
    selected_wellformed = wellformed[wellformed_idx]
    final_list = selected_illformed.tolist() + selected_wellformed.tolist()
    random.shuffle(final_list)
    langs = [lang] * len(final_list)
    
    with open(src_write_path, "a", encoding="utf-8") as fsrcout, \
         open(tgt_write_path, "a", encoding="utf-8") as ftgtout, \
         open(lang_write_path, "a", encoding="utf-8") as flangout:
            fsrcout.write("".join(list(zip(*final_list))[0]))
            ftgtout.write("".join(list(zip(*final_list))[1]))
            flangout.write("\n".join(langs) + "\n")
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
        write_paths = UpstreamPredictionPaths(lang="combined", pos="UPOS", split=split)
        src_write_path = write_paths.linearized(is_target=False)
        tgt_write_path = write_paths.linearized(is_target=True)
        lang_write_path = write_paths.language_file()
        # src_write_path = get_combined_txt_path(split, is_target=False)
        # tgt_write_path = get_combined_txt_path(split, is_target=True)
        # lang_write_path = get_combined_langs_txt_path(split)

        with open(src_write_path, "w", encoding="utf-8") as fsrcout, \
             open(tgt_write_path, "w", encoding="utf-8") as ftgtout, \
             open(lang_write_path, "w", encoding="utf-8") as ftgtout:
                pass
           

    for lang in langs:
        train_count = 0
        dev_count = 0

        read_paths = UpstreamPredictionPaths(lang=lang, pos="UPOS", split=split)

        for split in splits:
            src_path = read_paths.linearized(is_target=False)
            tgt_path = read_paths.linearized(is_target=True)
            # src_path = get_linearized_txt_path(lang=lang, pos="upos", split=split, is_target=False)
            # tgt_path = get_linearized_txt_path(lang=lang, pos="upos", split=split, is_target=True)
            print(f"Loading from {src_path} and {tgt_path}...")

            # src_write_path = get_combined_txt_path(split, is_target=False)
            # tgt_write_path = get_combined_txt_path(split, is_target=True)
            # lang_write_path = get_combined_langs_txt_path(split)

           
            if split == "train":
                
                lang_train_count = combine_train_sentences(src_path, tgt_path, src_write_path, tgt_write_path, lang_write_path)
                expected_dev_count = round(lang_train_count/0.8*0.2)
                train_count += lang_train_count
                total_train_count += train_count
               
            elif split == "dev":
                # lang_write_path = get_combined_langs_txt_path("dev")
                lang_dev_count = combine_dev_sentences(src_path, tgt_path, src_write_path, tgt_write_path, lang_write_path, dev_count=expected_dev_count)
                dev_count += lang_dev_count
                total_dev_count += dev_count
     
            
        print(f"Split: {split}\nTotal number of sentences in train dataset: {train_count}\nTotal number in dev dataset: {dev_count}")
    print(total_train_count, total_dev_count)
