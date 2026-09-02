from itertools import product
from conllu import parse_incr
from tqdm import tqdm

# from src.pathgen import get_deprojz_file_path, get_const2dep_file_path,  get_txt2mrg_file_path, get_matched_file_path, get_label_mrg_path
from src.pathgen import DataPaths, UpstreamPredictionPaths, FinalPredictionPaths
from src.data.common.conllu_io import rewrite_conllu
from src.data.common.postprocessing.mrg_to_conllu import mrg_to_conllu
from src.data.downstream.postprocessing.txt2mrg import txt2mrg

def remove_mismatched_sentences(read_system_path, read_gold_path, write_system_path, write_gold_path):

    mismatched_count = 0

    with open(write_system_path, "w", encoding="utf-8") as sysout, \
         open(write_gold_path, "w", encoding="utf-8") as goldout:
        pass
    with open(read_system_path, "r", encoding="utf-8") as sysin, \
         open(read_gold_path, "r", encoding="utf-8") as goldin: 
        for tokenlist1, tokenlist2 in tqdm(zip(parse_incr(sysin), parse_incr(goldin)), desc="Removing mismatched sentences:"):
            sys_text = tokenlist1.metadata["text"]
            gold_text = tokenlist2.metadata["text"]
            if sys_text != gold_text:
                mismatched_count += 1
            else:
                with open(write_system_path, "a", encoding="utf-8") as sysout,  \
                     open(write_gold_path, "a", encoding="utf-8") as goldout:
                    sysout.write(tokenlist1.serialize())
                    goldout.write(tokenlist2.serialize())
    return mismatched_count       

def postprocessing_pipeline(lang, pos="XPOS", epochs=20, subfolder="label_experiments", head=None, path=None):

    # def ensure_list(arg):
    #     return arg if isinstance(arg, list) else [arg]

    # lang_name, pos, epochs = map(
    #     ensure_list,
    #     (lang_name, pos, epochs)
    # )

    # if is_neural:
    #     paras = list(product(lang_name, pos, epochs))
    data_paths = DataPaths(lang=lang, split="test")
    # upstream_prediction_paths = UpstreamPredictionPaths(lang=lang, pos=pos, split="test", epochs=epochs)
    final_prediction_paths = FinalPredictionPaths(lang=lang, pos=pos, epochs=epochs, head=head, path=path)
   

    if subfolder == "neural":
        # data_paths = DataPaths(lang=lang, split="test")
       
        read_linearized_path = final_prediction_paths.linearized()
        read_source_path = final_prediction_paths.constituentized()
        write_path = final_prediction_paths.delinearized()
        txt2mrg(read_linearized_path, read_source_path, write_path)
        
        # for para in paras:
            # read_linearized_path, read_source_path, read_orig_path, write_path = get_txt2mrg_file_path(*para)
            
            
    # === Tree conversion ===
    # if head is not None or path is not None:
    #     paras = list(product(lang_name, pos, epochs, [is_neural]))
    # else:
    #     paras = list(product(lang_name, pos, epochs))
    
    # for para in paras:
    #         lang_name, split = para
    #         if head is not None or path is not None:
    #             read_tree_path = get_label_mrg_path(*para, head=head,path=path)
    #             read_orig_path = get_raw_conllu_path(lang, split=)
    #         else:
    read_orig_path = data_paths.raw()
    write_path = final_prediction_paths.conllu(subfolder=subfolder)
    read_tree_path = final_prediction_paths.delinearized() if subfolder == "neural" else final_prediction_paths.constituentized()
    print(f"Reading from {read_orig_path} and {read_tree_path}...")
    print(f"Writing into {write_path}...")
    mrg_to_conllu(lang, read_tree_path, read_orig_path, write_path)

    # === Deprojectivization ===
    # paras = list(product(lang_name, pos, epochs, [is_neural]))
    
    # for para in paras:
        # read_path, write_path = get_deprojz_file_path(*para)
    # read_path = final_prediction_paths.delinearized() if subfolder == "neural" else final_prediction_paths.constituentized(head=head, path=path)
    read_path = final_prediction_paths.conllu(subfolder=subfolder)
    write_path = final_prediction_paths.deprojectivized(subfolder="label_experiments")
    print(f"Reading from {read_path}...")
    print(f"Writing into {write_path}...")
    restore_arcs_count = rewrite_conllu(read_path, write_path, projz_mode=False, pseudo_filter=False, head=head, path=path)


    # === Remove mismatched sentences ===
    # paras = list(product(lang_name, model, bert, charlm, pretrain, epochs))
    # for para in paras:
    #     read_system_path, read_gold_path, write_system_path, write_gold_path = get_matched_file_path(*para)
    #     mismatched_count = remove_mismatched_sentences(read_system_path, read_gold_path, write_system_path, write_gold_path)
    #     print(f"Number of mismatched sentences: {mismatched_count}")


def main():
    postprocessing_pipeline()

if __name__ == "__main__":
    main()
