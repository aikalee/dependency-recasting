from itertools import product

from src.pathgen import get_projz_file_path, get_dep2const_file_path
from common.conllu_io import rewrite_conllu
from src.common.preprocessing.conllu_to_mrg import conllu_to_mrg


def common_preprocessing_pipeline(lang_name, split_name, pos):
    """
    The workflow of the pipeline:
    -> Non-projective sentences in UD CoNLL-U format 
    -> Pseudo-projective sentences in UD CoNLL-U format 
    -> Pseudo-projective sentences in Penn Treebank dependency tree format 

    Input: .conllu files
    Output: .mrg files
    """

    def ensure_list(arg):
        return arg if isinstance(arg, list) else [arg]

    lang_name, split_name, pos = map(ensure_list, (lang_name, split_name, pos))

    # === Projectivization ===
    paras = list(product(lang_name, split_name))
    
    for para in paras:
        read_path, write_path = get_projz_file_path(*para)
        rewrite_conllu(read_path, write_path, projz_mode=True)

    # === Tree conversion ===
    paras = list(product(lang_name, split_name, pos))
      
    for para in paras:
        read_path, write_path = get_dep2const_file_path(*para)
        conllu_to_mrg(read_path, write_path, pos[0])


def main():
    preprocessing_pipeline()

if __name__ == "__main__":
    main()



