from itertools import product
# from src.pathgen import get_txt2mrg_file_path, get_const2dep_file_path, get_deprojz_file_path
from src.pathgen import DataPaths, UpstreamPredictionPaths, FinalPredictionPaths
from src.data.common.conllu_io import rewrite_conllu
from src.data.common.postprocessing.mrg_to_conllu import mrg_to_conllu
from src.data.downstream.postprocessing.txt_to_mrg import txt_to_mrg
from src.data.downstream.postprocessing.json_to_txt import json_to_txt

def postprocessing_pipeline(lang, pos="XPOS", epochs=20, subfolder="rule_based", gate=None, head=None, path=None):

    if subfolder not in ["linearized", "structured_tokens", "rule_based", "label_experiments"]:
        raise ValueError("Must be 'linearized', 'structured_tokens', 'rule_based' or 'label_experiments'.")
     
    if subfolder == "structured_tokens" and gate is None:
        raise ValueError("gate should be True or False.")

    if subfolder == "label_experiments":
        if head is None and path is None:
            raise ValueError("Either head or path should not be None.")

    data_paths = DataPaths(lang=lang, split="test")
    final_prediction_paths = FinalPredictionPaths(lang=lang, pos=pos, subfolder=subfolder, epochs=epochs, gate=gate, head=head, path=path)

    # === Structured tokens to linearized ===
    if subfolder == "structured_tokens":
        read_path = final_prediction_paths.structured_tokens()
        write_path = final_prediction_paths.linearized()
        json_to_txt(read_path, write_path)
            
    # === Delinearization ===
    if subfolder in ["linearized", "structured_tokens"]:
        read_linearized_path = final_prediction_paths.linearized()
        read_source_path = final_prediction_paths.constituentized()
        read_orig_path = data_paths.raw()
        write_path = final_prediction_paths.delinearized()
        txt_to_mrg(read_linearized_path, read_source_path, read_orig_path, write_path)
    
    # === Tree conversion ===
    read_tree_path = final_prediction_paths.delinearized()
    read_orig_path = data_paths.raw()
    write_path = final_prediction_paths.conllu()
    mrg_to_conllu(lang, read_tree_path, read_orig_path, write_path)

    # === Deprojectivization ===
    read_path = final_prediction_paths.conllu()
    write_path = final_prediction_paths.deprojectivized()
    rewrite_conllu(read_path, write_path, False)


