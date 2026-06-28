import shutil
from itertools import product
from src.pathgen import get_constituentized_mrg_path, get_upstream_output_path, get_edit_actions_json_path, get_linearized_txt_path, get_lang_txt_path, get_structured_tokens_json_path
from src.data.downstream.preprocessing.mrg2txt import mrg2txt
from src.data.downstream.preprocessing.txt2json import txt_to_edit_actions_json, txt_to_structured_tokens_json

def downstream_preprocessing_pipeline(lang_name, split="train", pos="XPOS", epochs=20, overlap=0, is_target=False):

    def ensure_list(arg):
        return arg if isinstance(arg, list) else [arg]

    lang_name, split, pos, epochs = map(
        ensure_list,
        (lang_name, split, pos, epochs)
    )

  
    # === Linearization ===
    paras = list(product(lang_name, pos, split, epochs))
    
    for para in paras:
        cur_lang_name, cur_pos, cur_split, _ = para
        if cur_lang_name == "combined":
            continue
       
        # read_path, write_path = get_mrg2txt_file_path(*para)

        read_path = get_constituentized_mrg_path(cur_lang_name, cur_pos, cur_split) if is_target else get_upstream_output_path(*para)
        write_path = get_linearized_txt_path(cur_lang_name, cur_pos, cur_split, is_target)
        print(f"Loading from {read_path}...")
        print(f"Writing into {write_path}...")
        mrg2txt(read_path, write_path, add_bos=False)

    # === To edit actions ===
    # paras = list(product(lang_name, pos, split))
  
    # for para in paras:
       
    #     src_path = get_linearized_txt_path(*para, is_target=False)
    #     tgt_path = get_linearized_txt_path(*para, is_target=True)
    #     write_dir, write_path = get_edit_actions_json_path(*para)
    #     print(f"Loading from {src_path} and {tgt_path}...")
    #     print(f"Writing into {write_path}...")
    #     txt2json(src_path, tgt_path, write_path)
        # shutil.copy(str(src_path), str(write_dir))

    # === to token features ===
    paras = list(product(lang_name, pos, split))
  
    for para in paras:
       
        read_src_path = get_linearized_txt_path(*para, is_target=False)
        read_tgt_path = get_linearized_txt_path(*para, is_target=True)
        read_lang_path = get_lang_txt_path(*para) if lang_name[0] == "combined" else None
        write_path = get_structured_tokens_json_path(*para, overlap=overlap)
        # write_tgt_path = get_structured_tokens_json_path(*para, is_target=True)
        print(f"Loading from {read_src_path} and {read_tgt_path}...")
        print(f"Writing into {write_path}...")
        txt_to_structured_tokens_json(read_src_path, read_tgt_path, write_path, overlap=overlap, read_lang_path=read_lang_path)
      
    


    

    
    
       

