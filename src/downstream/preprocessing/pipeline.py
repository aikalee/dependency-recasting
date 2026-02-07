from itertools import product
from pathgen import get_mrg2txt_file_path
from src.downstream.preprocessing.mrg2txt import mrg2txt

def preprocessing_pipeline(lang_name, split="train", pos="XPOS", epochs=20, is_target=False):

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

    

    
    
       

