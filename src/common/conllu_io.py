from tqdm import tqdm

from collections import defaultdict, deque
from conllu import parse, parse_incr
from conllu.models import TokenList
from dataclasses import dataclass
from itertools import product
from typing import Optional

from src.common.preprocessing.projectivize import is_non_proj, get_non_proj_arcs, projectivize, relabel
from src.common.postprocessing.deprojectivize import deprojectivize_by_head, deprojectivize_by_path, is_projz


@dataclass
class SentenceData:

    arcs: list
    deprels: dict
    num_tokens: Optional[int] = None

    head_candidate_lookup: Optional[dict] = None
    path_candidate_lookup: Optional[dict] = None
    dlookup: Optional[dict] = None
    
    stack: Optional[deque] = None
    parent_stack: Optional[deque] = None

def init_sentencedata(deprels):

    arcs = list(deprels.keys())
    num_tokens = len(arcs)

    head_candidate_lookup = defaultdict(list) 
    path_candidate_lookup = defaultdict(list)
    dlookup = defaultdict(list)
    

    stack = deque()
    parent_stack = deque()

    for k, v in deprels.items():
            d, h = k
            dlookup[h] += [d] 
            head_candidate_lookup[(h, v)] += [d]
            
            if "↑" in v:
                stack += [k]
                arcs.remove(k)
            elif "↓" in v:
                path_candidate_lookup[h] += [d]
                parent_stack += [d]
  
                
    return SentenceData(
        arcs=arcs, 
        num_tokens=num_tokens, 
        deprels=deprels, 
        head_candidate_lookup=head_candidate_lookup, 
        path_candidate_lookup=path_candidate_lookup,
        dlookup=dlookup, 
        stack=stack,
        parent_stack=parent_stack
        )

def read_conllu(file_path):

    deprels = {}

    with open(file_path, "r", encoding="utf-8") as f:
       
        for tokenlist in parse_incr(f):
            deprels = {}
            tokens = []
           
            for token in tokenlist:
                
                if isinstance(token["id"], int):
                    deprels[(token["id"], token["head"])] = token["deprel"]
                    tokens.append(token)

            # no_mwt_tokenlist = TokenList(tokens, metadata=tokenlist.metadata)
            sentencedata = init_sentencedata(deprels)
            
                
            yield tokenlist, sentencedata

def read_conllu_sentence(sent):
    deprels = {}

    tokenlist = parse(sent)
    
    for token in tokenlist[0]:
        
        if isinstance(token["id"], int):
            deprels[(token["id"], token["head"])] = token["deprel"]

    sentencedata = init_sentencedata(deprels)
    
    return tokenlist[0], sentencedata

def reconstruct_conllu(tokenlist, transformed_deprels):

    if transformed_deprels:
        tokenlist = tokenlist.copy()

        # for k, v in transformed_deprels.keys():

        #     if tokenlist[int(k)-1]["id"] == k:
        #         tokenlist[int(k)-1]["head"] = v
        #         tokenlist[int(k)-1]["deprel"] = transformed_deprels[(k, v)]  
        #     else:
        #         raise ValueError("ID should be the same with k.")

        for idx, token in enumerate(tokenlist):
            for k, v in transformed_deprels.keys():
                if token["id"] == k:
                    tokenlist[idx]["head"] = v
                    tokenlist[idx]["deprel"] = transformed_deprels[(k, v)]
    return tokenlist


def rewrite_conllu(read_path, write_path, projz_mode=True, pseudo_filter=False) -> None:
    
    sents = read_conllu(read_path)
    desc = "Projectivizing" if projz_mode else "Deprojectivizing"

    with open(write_path, "w") as f:
        pass
    
    for tokenlist, sentencedata in tqdm(sents, desc=desc):

        arcs = sentencedata.arcs
        deprels = sentencedata.deprels
        dlookup = sentencedata.dlookup

        if projz_mode:
            if is_non_proj(arcs):
                projz_arcs = projectivize(arcs, symmetric_counting=True, dlookup=dlookup)
                projz_deprels = relabel(deprels, projz_arcs)      
                tokenlist = reconstruct_conllu(tokenlist, projz_deprels)    
        
            elif pseudo_filter:
                continue
        
        else:
            if is_projz(deprels):
                deprojz_deprels = deprojectivize_by_path(sentencedata)
                tokenlist = reconstruct_conllu(tokenlist, deprojz_deprels)
       
        conllu = tokenlist.serialize()        

        try:
            with open(write_path, "a", encoding="utf-8") as f:
                f.write(conllu)
                # print("Write sucessful:", tokenlist.metadata["sent_id"])
        except Exception as e:
            print("‼️ Write failed:", e)

def count_non_projectivity(read_path):
    all_non_proj_arcs = 0
    all_arcs = 0

    sents = read_conllu(read_path)
    for _, sentencedata in sents:
        arcs = sentencedata.arcs
        all_non_proj_arcs += len(get_non_proj_arcs(arcs))
        all_arcs += len(arcs)
    
    return all_non_proj_arcs / all_arcs * 100

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

def main():
    # ======================== Sample usage ====================================
    #    lang_name = ["Ancient_Greek", "English", "Danish", "Latin", "Old_East_Slavic", "Urdu"] 
    #     split_name = ["train", "dev", "test"]
    #     projz_flags = [True, False]
    #     pseudo_flags = [True, False]

    #     paras = list(product(lang_name, split_name, projz_flags, pseudo_flags))
        
    #     for para in paras:
    #         _, _, projz_flag, pseudo_flag = para
    #         read_path, write_path = construct_projz_file_path(*para)
    #         rewrite_conllu(read_path, write_path, projz_mode=projz_flag, pseudo_filter=pseudo_flag)
    # ==========================================================================

    # lang_name = ["English"]
    # split_name = ["train", "dev", "test"]
    # projz_flags = [True]
    # pseudo_flags = [True, False]

    # paras = list(product(lang_name, split_name, projz_flags, pseudo_flags))
    
    # for para in paras:
    #     _, _, projz_flag, pseudo_flag = para
    #     read_path, write_path = construct_projz_file_path(*para)
    #     rewrite_conllu(read_path, write_path, projz_mode=projz_flag, pseudo_filter=pseudo_flag)

    read_path = "predictions/en-penn-ud,filter=none,method=most-crossed,pos=upos.conllu"
    write_path = "predictions/en-penn-ud,filter=none,method=most-crossed-deprojz,pos=upos.conllu"
    rewrite_conllu(read_path, write_path, False, False)

  
if __name__ == "__main__":
    main()
