import os
import sys
from itertools import product
from conllu import parse_incr
from tqdm import tqdm

from conllu_io import construct_projz_file_path, rewrite_conllu
from mrg_to_conllu import construct_conllu_file_path, mrg_to_conllu

def remove_mismatched_sentences(read_system_path, read_gold_path, write_system_path, write_gold_path):

    mismatched_count = 0

    with open(write_system_path, "w", encoding="utf-8") as sysout, \
         open(write_gold_path, "w", encoding="utf-8") as goldout:
        pass
    with open(read_system_path, "r", encoding="utf-8") as sysin, \
         open(read_gold_path, "r", encoding="utf-8") as goldin: 
        for tokenlist1, tokenlist2 in tqdm(zip(parse_incr(sysin), parse_incr(goldin)), desc="Removing mismatched sentences:"):
            sys_text = tokenlist1.metadata["text"]
            gold_text = tokenlist2.metadata["text"]
            if sys_text != gold_text:
                mismatched_count += 1
            else:
                with open(write_system_path, "a", encoding="utf-8") as sysout,  \
                     open(write_gold_path, "a", encoding="utf-8") as goldout:
                    sysout.write(tokenlist1.serialize())
                    goldout.write(tokenlist2.serialize())
    return mismatched_count

def construct_matched_file_path(lang, epoch, pos="UPOS", pseudo_filter=False):

    # Input file name: en-penn-ud,filter=none,method=most-crossed-deprojz,pos=upos,epoch=20.conllu
    # Onput file name: en-penn-ud,filter=none,method=most-crossed-deprojz,pos=upos,epoch=20,matched=yes.conllu
    # language code reference: https://github.com/stanfordnlp/stanza/blob/dev/stanza/models/common/constant.py

    
    ud_abbr_lookup = {
        "Ancient_Greek": "grc",
        "Chinese": "zh",
        "Danish": "da",
        "English": "en",
        "Korean": "ko",
        "Latin": "la",
        "Old_East_Slavic": "orv",
        "Urdu": "ur"
    }

    ptb_abbr_lookup = {
        "Ancient_Greek": "grc",
        "Chinese": "zh-hant",
        "Danish": "da",
        "English": "en",
        "Korean": "ko",
        "Latin": "la",
        "Old_East_Slavic": "orv",
        "Urdu": "ur"
    }

    treebank_lookup = {
        "Ancient_Greek": "Perseus",
        "Chinese": "GSD",
        "Danish": "DDT",
        "English": "Penn",
        "Korean": "GSD",
        "Latin": "UDante",
        "Old_East_Slavic": "RNC",
        "Urdu": "UDTB"

    }

    ud_abbr = ud_abbr_lookup[lang]
    ptb_abbr = ptb_abbr_lookup[lang]
    treebank = treebank_lookup[lang]

    projz = "most-crossed"
    pos = pos.lower()
    pseudo = "pseudo" if pseudo_filter else "none"

    read_system_path = fr"predictions/{ptb_abbr}-{treebank.lower()}-ud,filter={pseudo},method={projz}-deprojz,pos={pos},epoch={epoch}.conllu"
    read_gold_path = fr"data/raw/UD_{lang}-{treebank}/{ud_abbr}_{treebank.lower()}-ud-test.conllu"
    write_system_path = fr"predictions/{ptb_abbr}-{treebank.lower()}-ud,filter={pseudo},method={projz}-deprojz,pos={pos},epoch={epoch},matched=yes.conllu"
    write_gold_path = fr"data/processed/gold/UD_{lang}-{treebank}/{ud_abbr}_{treebank.lower()}-ud-test,epoch={epoch},matched=yes.conllu"
            
    for read_path in [read_system_path, read_gold_path]:
        if not os.path.exists(read_path):
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
        
    output_dir = f"data/processed/gold/UD_{lang}-{treebank}/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created: {output_dir}")

    print(f"Loading from {read_system_path} and {read_gold_path}...")
    print(f"Writing into {write_system_path} and {write_gold_path}...")
    
    return read_system_path, read_gold_path, write_system_path, write_gold_path

           

def postprocessing_pipeline():

    # lang_name = ["Ancient_Greek", "Danish", "English", "Latin", "Old_East_Slavic", "Urdu"] 
    lang_name = ["English"]
    projz_flags = [False]
    pseudo_flags = [False]
    pos = ["UPOS"]
    epoch = ["20", "100"]

    # === Tree conversion ===
    paras = list(product(lang_name, epoch, pos, pseudo_flags))
    
    for para in paras:
        read_path, write_path = construct_conllu_file_path(*para)
        mrg_to_conllu(read_path, write_path)

    # === Deprojectivization ===
    paras = list(product(lang_name, epoch, projz_flags, pseudo_flags))
    
    for para in paras:
        read_path, write_path = construct_projz_file_path(*para)
        rewrite_conllu(read_path, write_path)

    # === Remove mismatched sentences ===
    paras = list(product(lang_name, epoch))
    for para in paras:
        read_system_path, read_gold_path, write_system_path, write_gold_path = construct_matched_file_path(*para)
        mismatched_count = remove_mismatched_sentences(read_system_path, read_gold_path, write_system_path, write_gold_path)
        print(f"Number of mismatched sentences: {mismatched_count}")


def main():
    postprocessing_pipeline()

if __name__ == "__main__":
    main()
