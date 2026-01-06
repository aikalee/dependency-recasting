from itertools import product

from src.pathgen import get_projz_file_path, get_tree_file_path
from common.conllu_io import rewrite_conllu
from src.preprocessing.conllu_to_mrg import conllu_to_mrg


def preprocessing_pipeline(lang_name, split_name):
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

    lang_name, split_name = map(ensure_list, (lang_name, split_name))

    # === Projectivization ===
    paras = list(product(lang_name, split_name))
    
    for para in paras:
        read_path, write_path = get_projz_file_path(*para)
        rewrite_conllu(read_path, write_path, projz_mode=True)

    # === Tree conversion ===
    paras = list(product(lang_name, split_name))
      
    for para in paras:
        read_path, write_path = get_tree_file_path(*para)
        conllu_to_mrg(read_path, write_path)


def main():
    preprocessing_pipeline()

if __name__ == "__main__":
    main()



