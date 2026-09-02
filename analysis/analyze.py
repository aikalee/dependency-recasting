from nltk.tree import Tree
from src.data.common.conllu_io import read_conllu
from src.data.common.preprocessing.projectivize import is_non_proj, get_non_proj_arcs
from src.data.common.postprocessing.deprojectivize import deprojectivize_by_path

def count_recovery_rate(orig_path, projz_path):
    all_arcs = 0
    all_non_proj_arcs_count = 0
    all_restore_arcs_count = 0
    projz_sents = list(read_conllu(projz_path))
    orig_sents = list(read_conllu(orig_path))

    if len(projz_sents) != len(orig_sents):
        raise ValueError(f"Number of sentences not match {len(projz_sents)}, {len(orig_sents)}.")
    for (proj_tokenlist, projz_sd), (orig_tokenlist, orig_sd) in zip(projz_sents, orig_sents):

        orig_sent_id = orig_tokenlist.metadata["sent_id"]
        proj_sent_id = proj_tokenlist.metadata["sent_id"]

        if orig_sent_id != proj_sent_id:
            raise ValueError("send_id not match.")
        

        # orig_dlookup = orig_sd.dlookup
        orig_arcs = orig_sd.arcs
        # orig_arcs = get_non_proj_arcs(orig_sd.arcs, symmetric_counting=True, dlookup=orig_dlookup)
        # orig_arcs = list(orig_arcs.keys())
        # if len(orig_arcs) == 0:
        #     continue
        _, restore_arcs = deprojectivize_by_path(projz_sd)
        # print(_)
        # print(restore_arcs)
        restore_arcs = list(restore_arcs.keys())
       
        # print(list(zip(*orig_arcs)))
        # for rd, rh in restore_arcs:
        all_od = list(zip(*orig_arcs))[0]
        if len(restore_arcs) != 0:
            all_restore_arcs = list(zip(*restore_arcs))[0]
        else:
            all_restore_arcs = []

        all_arcs += len(orig_arcs)
        for rd, rh in restore_arcs:
            # all_non_proj_arcs_count += 1
            for od, oh in orig_arcs:
                if rd not in all_od:
                    raise ValueError(f"Child not in list of non-projective arcs in sentence {orig_sent_id}: {rd}")
                if rd == od:
                    if oh == rh:
                        all_restore_arcs_count += 1
        
        for od, oh in orig_arcs:
            if od not in all_restore_arcs:
                all_restore_arcs_count += 1
            
                

    return all_restore_arcs_count / all_arcs * 100

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

def validate_all_labels(read_path):
    """
    Read .mrg file
    Check if there is only one parent label in the lifted arcs
    """
    def get_subtree_label(tree):
        for child in tree:
            label = child.label()
            if isinstance(child[0], str):
                return None
            elif "↑" in label and len(label.split("↑")) > 2:
                raise ValueError("Too many ancestor labels.")
            else:
                get_subtree_label(child)

    with open(read_path, "r", encoding="utf-8") as f:
        for line in f:
            tree = Tree.fromstring(line)
            get_subtree_label(tree)
        print("Done! No labels with multiple ancestors found.")

        
