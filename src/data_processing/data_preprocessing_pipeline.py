import os
import sys
from itertools import product

from conllu_io import rewrite_conllu, construct_projz_file_path
from conllu_to_mrg import conllu_to_mrg, construct_tree_file_path


def preprocessing_pipeline():
    """
    The workflow of the pipeline:
    -> Non-projective sentences in UD CoNLL-U format 
    -> Pseudo-projective sentences in UD CoNLL-U format 
    -> Pseudo-projective sentences in Penn Treebank dependency tree format 

    Input: .conllu files
    Output: .mrg files
    """

    # lang_name = ["Ancient_Greek", "Danish", "English", "Latin", "Old_East_Slavic", "Urdu"] 
    lang_name = ["English"]
    split_name = ["train", "dev", "test"]
    projz_flags = [True]
    pseudo_flags = [False, True]

    pos = ["UPOS"]

    # === Projectivization ===
    paras = list(product(lang_name, split_name, projz_flags, pseudo_flags))
    
    for para in paras:
        _, _, projz_flag, pseudo_flag = para
        read_path, write_path = construct_projz_file_path(*para)
        rewrite_conllu(read_path, write_path, projz_mode=projz_flag, pseudo_filter=pseudo_flag)

    # === Tree conversion ===
    paras = list(product(lang_name, split_name, pos, pseudo_flags))
      
    for para in paras:
        read_path, write_path = construct_tree_file_path(*para)
        conllu_to_mrg(read_path, write_path)


def main():
    preprocessing_pipeline()

if __name__ == "__main__":
    main()



