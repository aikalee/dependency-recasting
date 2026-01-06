from src.common.conllu_io import read_conllu
from src.common.preprocessing.dep2const import sentence2tree
from itertools import product
from tqdm import tqdm

def conllu_to_mrg(read_path, write_path):

    with open(write_path, "w", encoding="utf-8") as f:
        pass

    for tokenlist, sentencedata in tqdm(read_conllu(read_path), desc="Converting sentences to trees"):
        ptb = sentence2tree(sentencedata, tokenlist)
        with open(write_path, "a", encoding="utf-8") as f:
            f.write(ptb.pformat(margin=float('inf')) + "\n")

def main():
    # lang_name = ["Chinese"] 
    # split_name = ["train", "dev", "test"]
    # pos = ["UPOS"]
    # pseudo_flags = [True, False]

    # paras = list(product(lang_name, split_name, pos, pseudo_flags))
    
        
    # for para in paras:
    #     # _, _, pseudo_flag = para
    #     read_path, write_path = construct_tree_file_path(*para)
    #     write_mrg(read_path, write_path)
    read_path = "data/debug.conllu"
    write_path = "data/debug.output.mrg"
    conllu_to_mrg(read_path, write_path)

if __name__ == "__main__":
    main()
  