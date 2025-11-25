import os
import sys
from tqdm import tqdm

from collections import defaultdict, deque
from conllu import parse, parse_incr
from conllu.models import TokenList
from dataclasses import dataclass
from itertools import product
from typing import Optional

from projectivize import is_non_proj, get_non_proj_arcs, projectivize, relabel
from deprojectivize import deprojectivize_by_head, deprojectivize_by_path, is_projz


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

def initialize(deprels, projz_mode=True, head=False, path=True):

    arcs = list(deprels.keys())
    num_tokens = len(arcs)

    head_candidate_lookup = defaultdict(list) 
    path_candidate_lookup = defaultdict(list)
    dlookup = defaultdict(list)
    

    stack = deque()
    parent_stack = deque()

    for k, v in deprels.items():
            d, h = k
            
            if "↑" in v:
                stack += [k]
                arcs.remove(k)
            else:
                dlookup[h] += [d] 
                # if head:
                head_candidate_lookup[(h, v)] += [d]
                if path:
                    if "↓" in v:
                        # print(k, v)
                        # print(path_candidate_lookup)
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

            no_mwt_tokenlist = TokenList(tokens, metadata=tokenlist.metadata)
            sentencedata = initialize(deprels)
            
                
            yield no_mwt_tokenlist, sentencedata

def read_conllu_sentence(sent):
    deprels = {}

    tokenlist = parse(sent)
    
    for token in tokenlist[0]:
        
        if isinstance(token["id"], int):
            deprels[(token["id"], token["head"])] = token["deprel"]

    sentencedata = initialize(deprels)
    
    return tokenlist[0], sentencedata

def reconstruct_conllu(tokenlist, transformed_deprels):

    if transformed_deprels:
        tokenlist = tokenlist.copy()

        for k, v in transformed_deprels.keys():

            if tokenlist[int(k)-1]["id"] == k:
                tokenlist[int(k)-1]["head"] = v
                tokenlist[int(k)-1]["deprel"] = transformed_deprels[(k, v)]  
            else:
                raise ValueError("ID should be the same with k.")
    
    return tokenlist


def rewrite_conllu(read_path, write_path, projz_mode=True, pseudo_filter=False) -> None:
    
    sents = read_conllu(read_path)

    with open(write_path, "w") as f:
        pass

    for tokenlist, sentencedata in tqdm(sents, desc="Projectivizing/Deprojectivizing"):

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

def construct_projz_file_path(lang, epoch, projz_mode=False, pseudo_filter=False, split=""):

    # file name: lang_(method)_split_(pseudo)_(deprojz)

    if projz_mode and split not in ["train", "test", "dev"]:
        raise ValueError("The variable split must match one of the followings: train, test, dev")
    
    # projz_mode = False if split == "output" else projz_mode

    ud_abbr_lookup = {
        "Ancient_Greek": "grc",
        "Chinese": "zh",
        "Danish": "da",
        "English": "en",
        "Korean": "ko",
        "Latin": "la",
        "Old_East_Slavic": "orv",
        "Urdu": "ur"
    }

    # language code reference: https://github.com/stanfordnlp/stanza/blob/dev/stanza/models/common/constant.py
    ptb_abbr_lookup = {
        "Ancient_Greek": "grc",
        "Chinese": "zh-hant",
        "Danish": "da",
        "English": "en",
        "Korean": "ko",
        "Latin": "la",
        "Old_East_Slavic": "orv",
        "Urdu": "ur"
    }

    treebank_lookup = {
        "Ancient_Greek": "Perseus",
        "Chinese": "GSD",
        "Danish": "DDT",
        "English": "Penn",
        "Korean": "GSD",
        "Latin": "UDante",
        "Old_East_Slavic": "RNC",
        "Urdu": "UDTB"
    }

    ud_abbr = ud_abbr_lookup[lang]
    ptb_abbr = ptb_abbr_lookup[lang]
    treebank = treebank_lookup[lang]

    if projz_mode:
        input_file = "raw"
        output_file = "processed"
        projz = f"{treebank.lower()}-ud-"

        if pseudo_filter:
            pseudo_out = "_pseudo"

        else:
            pseudo_out = ""

        
    else:
        projz = "most-crossed"

        input_file = "baseline_outputs" if split == "output" else "processed"
        output_file = input_file

        pseudo = "pseudo" if pseudo_filter else "none"
        pos = "upos"
    
    output_dir = f"data/{output_file}/ud/UD_{lang}-{treebank}"
   
    # read_dir = f"../data/{input_dir}/UD_{lang}-{treebank}"
    if projz_mode:
        read_path = fr"data/{input_file}/UD_{lang}-{treebank}/{ud_abbr}_{projz}{split}.conllu"
        write_path = f"{output_dir}/{ud_abbr}_most-crossed_{split}{pseudo_out}.conllu"
       
    else:   
        read_path = fr"predictions/{ptb_abbr}-{treebank.lower()}-ud,filter={pseudo},method=most-crossed,pos={pos},epoch={epoch}.conllu"
        write_path = f"predictions/{ptb_abbr}-{treebank.lower()}-ud,filter={pseudo},method=most-crossed-deprojz,pos={pos},epoch={epoch}.conllu"
    
    
    if not os.path.exists(read_path):
            raise FileNotFoundError(f"The file '{read_path}' does not exist.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created: {output_dir}")
  
    print(f"Loading from {read_path}...")
    print(f"Writing into {write_path}...")
    
    return read_path, write_path
    
    
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
