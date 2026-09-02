from conllu import parse_incr
from itertools import product
from tqdm import tqdm
from src.pathgen import DataPaths

def replace_bracket(read_path, write_path):
    with open(write_path, "w") as fout:
        pass
    with open(read_path, encoding="utf-8") as fin:
        for tokenlist in tqdm(parse_incr(fin), desc="Replacing brackets"):
            for idx, token in enumerate(tokenlist):
                if token["form"] in ["(", "（"]:
                    tokenlist[idx]["form"] = "-LRB-"
                elif token["form"] in [")", "）"]:
                    tokenlist[idx]["form"] = "-RRB-"
        
            conllu = tokenlist.serialize()   
            with open(write_path, "a", encoding="utf-8") as fout:
                fout.write(conllu)

def replace_bracket_upstream_inference(lang, split):
    
    def ensure_list(arg):
        return arg if isinstance(arg, list) else [arg]

    lang_name, split_name = map(ensure_list, (lang_name, split_name))
    paras = list(product(lang_name, split_name))
    
    for para in paras:
        # read_path, write_path = get_upsteam_inference_file_path(*para)
        data_paths = DataPaths(lang=lang, split=split)
        read_path = data_paths.raw()
        write_path = data_paths.upstream()
        replace_bracket(read_path, write_path)
