from itertools import product
from pathgen import get_linearize_file_path
from src.postprocessing.linearize import linearize
from src.postprocessing.mrg2txt import mrg2txt

def postprocessing_pipeline(lang_name, model, bert="frozen", charlm="yes", pretrain="yes", epochs=20):

    def ensure_list(arg):
        return arg if isinstance(arg, list) else [arg]

    lang_name, model, bert, charlm, pretrain, epochs = map(
        ensure_list,
        (lang_name, model, bert, charlm, pretrain, epochs)
    )

    # === Linearization ===
    paras = list(product(lang_name, model, bert, charlm, pretrain, epochs))
    
    for para in paras:
        read_path, write_path = get_linearize_file_path(*para)
        mrg2txt(read_path, write_path)

    

    
    
       

