from conllu import TokenList, parse_incr
from nltk import Tree
from tqdm import tqdm
from src.downstream.postprocessing.delinearize import validate_linearized_brackets, linearized_to_ptb

def count_linearized(linearized):
    return len([t for t in linearized.strip().split() if t.isalpha() and t.isupper()])

def count_source(source):
    return len(Tree.fromstring(source).leaves()) 

def txt2mrg(read_linearized_path, read_source_path, read_orig_path, write_path):
    sent_id = 1
    count_illformed = 0
    with open(write_path, "w", encoding="utf-8") as fout:
        pass
    with open(read_linearized_path, "r", encoding="utf-8") as flin,\
         open(read_source_path, "r", encoding="utf-8") as fsrc, \
         open(read_orig_path, "r", encoding="utf-8") as forig:
        for linearized, src, orig in tqdm(zip(flin, fsrc, parse_incr(forig)), desc="Delinearizing"):
            equal_words = count_linearized(linearized) == count_source(src)
            is_validate, msg = validate_linearized_brackets(linearized)
            if is_validate and equal_words:
                delinearized = linearized_to_ptb(linearized) 
            else:
                delinearized = src
                count_illformed += 1
                print(msg + f"in sentence {sent_id}.")
            sent_id += 1
            with open(write_path, "a", encoding="utf-8") as fout:
                fout.write(delinearized.strip() + "\n")
    print(f"There are {count_illformed} illformed sentences.")
