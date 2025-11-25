import os
import sys
from nltk.tree import Tree
from tqdm import tqdm

from data_loader import sentence_to_conllu
from dependency2constituency import tree2sentence

def mrg_to_conllu(read_path, write_path):
    illformed_count = 0
    total_count = 0
    sent_id = 1

    with open(write_path, "w", encoding="utf-8") as fout:
        pass

    with open(read_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Converting trees to sentences"):
            sentence, is_illformed = tree2sentence(Tree.fromstring(line))
            conllu = sentence_to_conllu(sentence, sent_id)
             
            sent_id += 1 
            total_count += 1

            if is_illformed:
                illformed_count += 1

            with open(write_path, "a", encoding="utf-8") as fout:
                fout.write(conllu + "\n")
    print(f"Pecentage of illformed sentences: {illformed_count/total_count:.2%}")

def construct_conllu_file_path(lang, epoch, pos="UPOS", pseudo_filter=True):

    # Input file name: lang=en-penn-ud,filter=none,method=most-crossed,pos=upos.mrg
    # Onput file name: lang=en-penn-ud,filter=none,method=most-crossed,pos=upos.conllu
    # language code reference: https://github.com/stanfordnlp/stanza/blob/dev/stanza/models/common/constant.py
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

    ptb_abbr = ptb_abbr_lookup[lang]
    treebank = treebank_lookup[lang].lower()

    projz = "most-crossed"
    pos = pos.lower()
    pseudo = "pseudo" if pseudo_filter else "none"

    read_path = fr"predictions/{ptb_abbr}-{treebank}-ud,filter={pseudo},method={projz},pos={pos},epoch={epoch}.mrg"
    write_path = fr"predictions/{ptb_abbr}-{treebank}-ud,filter={pseudo},method={projz},pos={pos},epoch={epoch}.conllu"
            
    if not os.path.exists(read_path):
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
    # if not os.path.exists(write_path):
    #     os.makedirs(write_path, exist_ok=True)
    #     print(f"Created: {write_path}")

    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path


def main():
    read_path = "predictions/en-penn-ud,filter=none,method=most-crossed,pos=upos,epoch=20.mrg"
    write_path = "predictions/en-penn-ud,filter=none,method=most-crossed,pos=upos,epoch=20conllu"
    mrg_to_conllu(read_path, write_path)

if __name__ == "__main__":
    main()



 

    


