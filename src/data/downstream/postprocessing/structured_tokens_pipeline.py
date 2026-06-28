from itertools import product
from src.pathgen import get_predicted_structured_tokens_json_path, get_linearized_structured_tokens_txt_path, get_predicted_mrg_path, get_raw_conllu_path, get_delinearized_structured_tokens_mrg_path, get_structured_tokens_conllu_path, get_deprojectivized_structured_tokens_conllu_path
from src.data.common.conllu_io import rewrite_conllu
from src.data.common.postprocessing.mrg_to_conllu import mrg_to_conllu
from src.data.downstream.postprocessing.txt2mrg import txt2mrg
from src.data.downstream.postprocessing.json2txt import json2txt

def structured_tokens_postprocessing_pipeline(lang_name, pos="upos", epochs=100, gate="yes"):

    def ensure_list(arg):
        return arg if isinstance(arg, list) else [arg]

    # lang_name = map(
    #     ensure_list,
    #     lang_name
    # )
    lang_name = ensure_list(lang_name)

    # === Structured tokens to lineazied ===
    # paras = list(product(lang_name, pos, gate))

    # for para in paras:
    for lang in lang_name:
        print(lang)
        read_path = get_predicted_structured_tokens_json_path(lang, pos=pos, gate=gate)
        write_path = get_linearized_structured_tokens_txt_path(lang, pos=pos, gate=gate)
        json2txt(read_path, write_path)
            
    # === Delinearization === 
    for lang in lang_name:
        read_linearized_path = get_linearized_structured_tokens_txt_path(lang, pos=pos, gate=gate)
        read_source_path = get_predicted_mrg_path(lang, pos=pos, epochs=epochs)
        read_orig_path = get_raw_conllu_path(lang, split="test")
        write_path = get_delinearized_structured_tokens_mrg_path(lang, pos=pos, gate=gate)
        txt2mrg(read_linearized_path, read_source_path, read_orig_path, write_path)
    
    # === Tree conversion ===
    # paras = list(product(lang_name, pos, epochs, is_neural))
    
    for lang in lang_name:
        read_tree_path = get_predicted_mrg_path(lang, pos=pos, epochs=epochs)
        read_orig_path = get_raw_conllu_path(lang, split="test")
        write_path = get_structured_tokens_conllu_path(lang, pos=pos, gate=gate)
        mrg_to_conllu(lang_name, read_tree_path, read_orig_path, write_path)

    # === Deprojectivization ===
    # paras = list(product(lang_name, pos, epochs, is_neural))
    
    for lang in lang_name:
        read_path =  get_structured_tokens_conllu_path(lang, pos=pos, gate=gate)
        write_path = get_deprojectivized_structured_tokens_conllu_path(lang, pos=pos, gate=gate)
        rewrite_conllu(read_path, write_path, False)
