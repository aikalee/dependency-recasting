from itertools import product
from src.pathgen import get_txt2mrg_file_path, get_const2dep_file_path, get_deprojz_file_path
from src.data.common.conllu_io import rewrite_conllu
from src.data.common.postprocessing.mrg_to_conllu import mrg_to_conllu
from src.data.downstream.postprocessing.txt2mrg import txt2mrg
from src.data.downstream.postprocessing.json2txt import json2txt

def postprocessing_pipeline(lang_name, pos="XPOS", epochs=20, is_neural=True):

    def ensure_list(arg):
        return arg if isinstance(arg, list) else [arg]

    lang_name, pos, epochs, is_neural = map(
        ensure_list,
        (lang_name, pos, epochs, is_neural)
    )

            
    # === Delinearization ===
    if is_neural:
        paras = list(product(lang_name, pos, epochs))
        
        for para in paras:
            read_linearized_path, read_source_path, read_orig_path, write_path = get_txt2mrg_file_path(*para)
            txt2mrg(read_linearized_path, read_source_path, read_orig_path, write_path)
    
    # === Tree conversion ===
    paras = list(product(lang_name, pos, epochs, is_neural))
    
    for para in paras:
        read_tree_path, read_orig_path, write_path = get_const2dep_file_path(*para)
        mrg_to_conllu(lang_name, read_tree_path, read_orig_path, write_path)

    # === Deprojectivization ===
    paras = list(product(lang_name, pos, epochs, is_neural))
    
    for para in paras:
        read_path, write_path = get_deprojz_file_path(*para)
        rewrite_conllu(read_path, write_path, False)


