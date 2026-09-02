from itertools import product

# from src.pathgen import get_raw_conllu_path, get_projectivized_conllu_path, get_dep2const_file_path, get_constituentized_mrg_path, get_projectivized_label_path, get_constituentized_label_path, get_add_to_dev_path, get_add_to_train_path
from src.pathgen import DataPaths
from src.data.common.conllu_io import rewrite_conllu
from src.data.common.preprocessing.conllu_to_mrg import conllu_to_mrg
from src.data.common.preprocessing.add_examples import add_examples

def ensure_list(arg):
    return arg if isinstance(arg, list) else [arg]

def common_preprocessing_pipeline(lang, split, pos, head=None, path=None, labels_aligned=True):
    """
    The workflow of the pipeline:
    -> Non-projective sentences in UD CoNLL-U format 
    -> Pseudo-projective sentences in UD CoNLL-U format 
    -> Pseudo-projective sentences in Penn Treebank dependency tree format 

    Input: .conllu files
    Output: .mrg files
    """

    # lang_name, split_name = map(ensure_list, (lang_name, split_name))

    # === Projectivization ===
    # paras = list(product(lang_name, split_name))

    data_paths = DataPaths(lang=lang, split=split)
    
    # for para in paras:
    # if head is not None or path is not None:
    
    if not labels_aligned:
        read_path = data_paths.raw()
        write_path = data_paths.projectivized(head=head, path=path)
    
            # read_path = get_raw_conllu_path(*para)
            # write_path = get_projectivized_conllu_path(*para)
        print(f"Loading from {read_path}")
        print(f"Writing into {write_path}")
        rewrite_conllu(read_path, write_path, projz_mode=True, head=head, path=path)

    # === Tree conversion ===
    # para_1 = list(product(lang_name, split_name, pos))
      
    # for para in paras:
    #     if head is not None or path is not None:
    #         read_path = get_projectivized_label_path(*para, head, path)
    #         write_path = get_constituentized_label_path(*para, pos, head, path)
    #     else:
    #         read_path = get_projectivized_conllu_path(*para, head, path)
    #         write_path = get_constituentized_mrg_path(*para, pos, head, path)
    read_path = data_paths.projectivized(head=head, path=path)
    write_path = data_paths.constituentized(pos=pos, head=head, path=path)
    print(f"Loading from {read_path}")
    print(f"Writing into {write_path}")
    # read_path = get_projectivized_conllu_path(*para)
    # read_path, write_path = get_dep2const_file_path(*para)
    conllu_to_mrg(read_path, write_path, pos)

def balance_label_coverage(lang, pos, head=None, path=None):
   
   
    # for lang in lang_name:
    train_data_paths = DataPaths(lang=lang, split="train")
    dev_data_paths = DataPaths(lang=lang, split="dev")
    train_path = train_data_paths.projectivized(head=head, path=path)
    dev_path = dev_data_paths.projectivized(head=head, path=path)
    print(f"Reading and writing into {train_path} and {dev_path}....")

    train_record_path = train_data_paths.records(head=head, path=path)
    dev_record_path = dev_data_paths.records(head=head, path=path)

    # train_path = get_projectivized_label_path(lang, "train", head, path)
    # dev_path = get_projectivized_label_path(lang, "dev", head, path)
    # train_recording_path = get_add_to_train_path(lang, head, path)
    # dev_recording_path = get_add_to_dev_path(lang, head, path)
    add_examples(train_path, dev_path, train_record_path, dev_record_path)

    
    # split_name = ["train", "dev"]
    # lang_name, split_name = map(ensure_list, (lang_name, split_name))
    # paras = list(product(lang_name, split_name))
    # for para in paras:
    # if head is not None or path is not None:
    #     read_path = get_projectivized_label_path(*para, head, path)
    #     write_path = get_constituentized_label_path(*para, pos, head, path)
    # else:
    #     read_path = get_projectivized_conllu_path(*para, head, path)
    #     write_path = get_constituentized_mrg_path(*para, pos, head, path)

    for split in ["train", "dev"]:
        read_path = train_data_paths.projectivized(head=head, path=path)
        write_path = train_data_paths.constituentized(pos=pos, head=head, path=path)
        # read_path = get_projectivized_conllu_path(*para)
        # read_path, write_path = get_dep2const_file_path(*para)
        conllu_to_mrg(read_path, write_path, pos)


def main():
    common_preprocessing_pipeline()

if __name__ == "__main__":
    main()



