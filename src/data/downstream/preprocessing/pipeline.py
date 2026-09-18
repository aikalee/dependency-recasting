import shutil
from itertools import product
# from src.pathgen import get_constituentized_mrg_path, get_upstream_output_path, get_edit_actions_json_path, get_linearized_txt_path, get_lang_txt_path, get_structured_tokens_json_path
from src.pathgen import DataPaths, UpstreamPredictionPaths
from src.data.downstream.preprocessing.mrg_to_txt import mrg_to_txt
from src.data.downstream.preprocessing.txt_to_json import txt_to_edit_actions_json, txt_to_structured_tokens_json

def downstream_preprocessing_pipeline(lang, split="train", pos="XPOS", epochs=20, overlap=0, is_target=False):

    # === Linearization ===
    data_paths = DataPaths(lang=lang, split=split)
    upstream_prediction_paths = UpstreamPredictionPaths(lang=lang, pos=pos, split=split, epochs=epochs)
    read_path = data_paths.constituentized(pos=pos, head=None, path=None) if is_target else upstream_prediction_paths.upstream_output()
    write_path = upstream_prediction_paths.linearized(is_target=is_target)
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    mrg_to_txt(read_path, write_path, add_bos=False)

    # === To structured tokens ===
    read_src_path = upstream_prediction_paths.linearized(is_target=False)
    read_tgt_path = upstream_prediction_paths.linearized(is_target=True)
    read_lang_path = upstream_prediction_paths.language_file() 
    print(f"Loading from {read_src_path} and {read_tgt_path}...")
    print(f"Writing into {write_path}...")
    txt_to_structured_tokens_json(read_src_path, read_tgt_path, write_path, overlap=overlap, read_lang_path=read_lang_path)
    
    


    

    
    
       

