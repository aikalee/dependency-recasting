from src.common.preprocessing.projectivize import is_non_proj, get_non_proj_arcs
from src.common.conllu_io import read_conllu
from src.common.postprocessing.deprojectivize import deprojectivize_by_path

def count_deprojectivization(read_path):
    all_arcs_count = 0
    all_restore_arcs_count = 0
    sents = read_conllu(read_path)
    for _, sentencedata in sents:
        _, restore_arcs = deprojectivize_by_path(sentencedata)
        all_arcs_count += len(sentencedata.arcs)
        all_restore_arcs_count += len(restore_arcs)
    return all_restore_arcs_count / all_arcs_count * 100

def count_non_projectivity(read_path):
    all_non_proj_arcs = 0
    all_arcs = 0

    sents = read_conllu(read_path)
    for _, sentencedata in sents:
        arcs = sentencedata.arcs
        if is_non_proj(arcs):
            all_non_proj_arcs += len(get_non_proj_arcs(arcs))
        all_arcs += len(arcs)

    return all_non_proj_arcs, all_arcs, all_non_proj_arcs / all_arcs * 100

def count_deprels(read_path):
    all_deprels = []
    sents = read_conllu(read_path)
    for tokenlist, _ in sents:
        for token in tokenlist:
            all_deprels.append(token["deprel"])
    all_unique_deprels = set(all_deprels)
    return all_unique_deprels, len(all_deprels)

def get_train_deprels(read_path):
    all_deprels = []

    sents = read_conllu(read_path)
    for tokenlist, _ in sents:
        for token in tokenlist:
            if isinstance(token["id"], int):
                all_deprels.append(token["deprel"])
    return set(all_deprels)

def sentence_add_to_train(train_path, dev_path):
    add_to_train = []
    added_labels = []
    train_deprels = get_train_deprels(train_path)
    sents = read_conllu(dev_path)
    for i, (tokenlist, _) in enumerate(sents, start=1):
        sent_id = tokenlist.metadata["sent_id"]
        for token in tokenlist:
            if isinstance(token["id"], int):
                if token["deprel"] not in train_deprels and token["deprel"] not in added_labels:
                        added_labels.append(token["deprel"])
                        add_to_train.append((sent_id, i))
    return add_to_train, added_labels