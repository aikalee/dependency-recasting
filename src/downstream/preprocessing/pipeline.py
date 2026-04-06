from itertools import product
from pathgen import get_mrg2txt_file_path, get_edit_actions_txt_path, get_linearized_txt_path
from src.downstream.preprocessing.mrg2txt import mrg2txt
from src.downstream.preprocessing.txt2txt import txt2txt

def downstream_preprocessing_pipeline(lang_name, split="train", pos="XPOS", epochs=20, is_target=False):

    def ensure_list(arg):
        return arg if isinstance(arg, list) else [arg]

    lang_name, split, pos, epochs, is_target = map(
        ensure_list,
        (lang_name, split, pos, epochs, is_target)
    )

    # === Linearization ===
    paras = list(product(lang_name, split, pos, epochs, is_target))
    
    for para in paras:
        read_path, write_path = get_mrg2txt_file_path(*para)
        mrg2txt(read_path, write_path)
    
    # === To edit actions ===
    paras = list(product(lang_name, pos, split))

    for para in paras:
        src_path = get_linearized_txt_path(*para, is_target=False)
        tgt_path = get_linearized_txt_path(*para, is_target=True)
        write_path = get_edit_actions_txt_path(*para)
        print(f"Loading from {src_path} and {tgt_path}...")
        print(f"Writing into {write_path}...")
        txt2txt(src_path, tgt_path, write_path)

    

    
    
       

