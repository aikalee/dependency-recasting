from itertools import product
from conllu import parse_incr
from tqdm import tqdm

from pathgen import get_deprojz_file_path, get_const2dep_file_path,  get_txt2mrg_file_path, get_matched_file_path
from src.common.conllu_io import rewrite_conllu
from src.common.postprocessing.mrg_to_conllu import mrg_to_conllu
from src.downstream.postprocessing.txt2mrg import txt2mrg

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

def postprocessing_pipeline(lang_name, pos="XPOS", epochs=20, is_neural=False):

    def ensure_list(arg):
        return arg if isinstance(arg, list) else [arg]

    lang_name, pos, epochs = map(
        ensure_list,
        (lang_name, pos, epochs)
    )

    if is_neural:
        paras = list(product(lang_name, pos, epochs))
        
        for para in paras:
            read_linearized_path, read_source_path, read_orig_path, write_path = get_txt2mrg_file_path(*para)
            txt2mrg(read_linearized_path, read_source_path, read_orig_path, write_path)
            
    # === Tree conversion ===
    paras = list(product(lang_name, pos, epochs, [is_neural]))
    
    for para in paras:
        read_tree_path, read_orig_path, write_path = get_const2dep_file_path(*para)
        mrg_to_conllu(lang_name, read_tree_path, read_orig_path, write_path)

    # === Deprojectivization ===
    paras = list(product(lang_name, pos, epochs, [is_neural]))
    
    for para in paras:
        read_path, write_path = get_deprojz_file_path(*para)
        rewrite_conllu(read_path, write_path, False)

    # === Remove mismatched sentences ===
    # paras = list(product(lang_name, model, bert, charlm, pretrain, epochs))
    # for para in paras:
    #     read_system_path, read_gold_path, write_system_path, write_gold_path = get_matched_file_path(*para)
    #     mismatched_count = remove_mismatched_sentences(read_system_path, read_gold_path, write_system_path, write_gold_path)
    #     print(f"Number of mismatched sentences: {mismatched_count}")


def main():
    postprocessing_pipeline()

if __name__ == "__main__":
    main()
