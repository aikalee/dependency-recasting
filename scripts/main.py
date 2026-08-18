import os
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.data.upstream.replace_bracket import replace_bracket_upstream_inference
from src.data.common.preprocessing.pipeline import common_preprocessing_pipeline, add_to_dev_pipeline
from src.data.downstream.preprocessing.pipeline import downstream_preprocessing_pipeline
from src.data.downstream.preprocessing.replace_pos import replace_pos_downstream_preprocessing
from src.data.downstream.preprocessing.mrg2txt import mrg2txt
from src.data.common.postprocessing.pipeline import postprocessing_pipeline
from src.data.downstream.preprocessing.combine_datasets import combine_datasets
from src.data.downstream.preprocessing.build_vocab_from_datasets import write_to_json
from src.data.downstream.postprocessing.structured_tokens_pipeline import structured_tokens_postprocessing_pipeline

# === for debugging ===
from nltk.tree import Tree
from conllu import parse, parse_incr
from src.data.common.conllu_io import read_conllu, rewrite_conllu, reconstruct_conllu
from src.data.common.postprocessing.mrg_to_conllu import mrg_to_conllu
from src.data.downstream.postprocessing.txt2mrg import txt2mrg
from src.data.common.preprocessing.dep2const import sentence2tree
from src.data.common.preprocessing.projectivize import projectivize, relabel
from src.data.common.postprocessing.deprojectivize import deprojectivize_by_head, deprojectivize_by_path, deprojectivize_by_head_path

# === analysis ===
# from src.pathgen import get_upstream_conllu_path, get_projectivized_conllu_path
from analysis.analyze import count_recovery_rate


def main():
    """
    language options:  
        "Ancient_Greek": "grc",
        "Chinese": "zh-hans",
        "English-Penn": "en-penn",
        "English-EWT": "en-ewt",
        "Finnish": "fi",
        "French": "fr",
        "Hebrew": "he",
        "Russian": "ru",
        "Tamil": "ta",
        "Uyghur": "ug",
        "Wolof": "wo"
    """

    """
    Steps:
    1) Downstream preprocess combine_langs
    2) Combine datasets into one dataset and write vocab and weight to json
    3) Downstream preprocess combined dataset
    """
    combine_langs = ["Ancient_Greek", "English-EWT", "English-Penn", "Finnish", "French", "Hebrew", "Russian",  "Tamil",  "Uyghur", "Wolof"]
    # combine_datasets()
    # replace_bracket_upstream_inference("Ancient_Greek", ["train", "dev", "test"])
    head = True
    path = True
    # preprocessing -> add to dev -> preprocessing
    common_preprocessing_pipeline("Ancient_Greek", "test", "UPOS", head=head, path=path)
    # add_to_dev_pipeline("Ancient_Greek", "UPOS", head=head, path=path)
    # postprocessing_pipeline(lang="Ancient_Greek", pos="UPOS", epochs=100, subfolder="label_experiments", head=True, path=True)
    # downstream_preprocessing_pipeline("English-Penn", ["train", "dev", "test"], "UPOS", 100, is_target=True, overlap=3)
    # structured_tokens_postprocessing_pipeline(lang_name="English-Penn", pos="upos", epochs=100, gate="none")
    # downstream_preprocessing_pipeline(combine_langs, ["train", "dev", "test"], "UPOS", 100, is_target=True)
    # write_to_json(langs=combine_langs, max_weight=20)
    # preprocessing_pipeline("Ancient_Greek", ["train", "dev", "test"], "XPOS")
    # postprocessing_pipeline("Ancient_Greek", "upos", 100, is_neural=True)
    # preprocessing_pipeline("Chinese", ["train", "dev", "test"])
    # postprocessing_pipeline("Polish", "stanza", "finetune", "no", "yes", 100)
    # for split in ["train", "dev", "test"]:
    #     gold_const_file = ROOT / "data" / "common" / "constituentized" / "lang=wo,pos=upos" / f"wo__{split}.mrg"
    #     pred_const_file = ROOT / "data" / "downstream" / "upstream_outputs" / "lang=wo,pos=pred-upos" / f"lang=wo,split={split},pos=pred-upos,epochs=100.mrg"
    #     output_const_file = ROOT / "data" / "common" / "constituentized" / "lang=wo,pos=pred-upos" / f"wo__{split}.mrg"
    #     output_linearized_file = ROOT / "data" / "downstream" / "linearized"/ "lang=wo,pos=all-pred-upos" / f"{split}.tgt.txt"
    #     replace_pos_downstream_preprocessing(gold_const_file, pred_const_file, output_const_file)
    #     mrg2txt(output_const_file, output_linearized_file)

    # ------------ for linearization debugging -------------------
    # DATA = ROOT / "debug.conllu"
    # for tokenlist, sentencedata in read_conllu(DATA):
    #     arcs = sentencedata.arcs
    #     dlookup = sentencedata.dlookup
        
    #     deprels = sentencedata.deprels
    #     new_arcs = projectivize(arcs, symmetric_counting=False)
        
    #     new_labels = relabel(deprels, new_arcs)
    #     print(new_labels)
    #     tokenlist = reconstruct_conllu(tokenlist, new_labels)
    #     conllu = tokenlist.serialize()
    #     with open(ROOT / "debug.output.conllu", "w", encoding="utf-8") as fout:
    #         fout.write(conllu)
    # ------------------------------------------------------------

    # --------- for finding English non projective sentence ------

    # DATA = ROOT / "data" / "upstream" / "UD_French-GSD" / "fr_gsd-ud-train.conllu"

    # for tokenlist, sentencedata in read_conllu(DATA):
    #     arcs = sentencedata.arcs
    #     if is_non_proj(arcs):
    #         print(tokenlist.metadata["sent_id"])
    #         print(tokenlist)
            
    # ------------------------------------------------------------
        
    # txt2mrg(ROOT / "debug.txt", ROOT / "debug.mrg", ROOT / "debug.conllu", ROOT / "debug.output.mrg")
    # mrg_to_conllu("Finnish", ROOT / "debug.output.mrg", ROOT / "debug.conllu", ROOT / "debug.output.conllu")
    # rewrite_conllu(ROOT / "debug.output.conllu", ROOT / "debug.output.deprojz.conllu", False)
    # print(Tree.fromstring('(TOP (root (obj (NOUN Palon)) (VERB kerrotaan) (ccomp↓ (VERB saaneen) (obj (NOUN alkunsa)) (obl (acl (obl (nmod_poss (NOUN matkakeskuksen)) (NOUN pysäköintikerroksessa)) (VERB olleista)) (NOUN styroksilevyistä)) (conj (cc (CCONJ ja)) (VERB tuhonneen) (obj (amod (nmod_poss (nummod (advmod (ADV noin)) (NUM 5 000)) (NOUN neliömetrin)) (ADJ kokoisen)) (NOUN alueen)))) (punct (PUNCT .))))'))

    # for tokenlist, sentencedata in read_conllu(DATA):
    #     print(sentencedata.deprels)
    #     print(sentencedata.arcs)
    #     deprojz = sentence2tree(sentencedata, tokenlist)
    #     print(deprojz)

    # ---------------------- for analysis --------------------------
    # lang = "Wolof"
    # orig_path = get_upstream_conllu_path(lang=lang, split="train")
    # projz_path = get_projectivized_conllu_path(lang=lang, split="train")
    # recovered = count_recovery_rate(orig_path, projz_path)
    # print("Language:", lang, "Recovery rate:", round(recovered, 2))


    # ------------------ deprojectivization debugging ------------------

    # DATA = ROOT / "debug.conllu"
    # for tokenlist, sentencedata in read_conllu(DATA):
    #     arcs = sentencedata.arcs
    #     deprels = sentencedata.deprels
    #     dlookup = sentencedata.dlookup

    #     projz_arcs = projectivize(arcs, symmetric_counting=True, dlookup=dlookup)
    #     print(projz_arcs)
    #     projz_deprels = relabel(deprels, projz_arcs, head=head, path=path)  
    #     print(projz_deprels)
    #     tokenlist = reconstruct_conllu(tokenlist, projz_deprels)
    #     print(tokenlist)
        # head_path_deprojz = deprojectivize_by_head(sentencedata)
        # print(head_path_deprojz)
    # ---------------------------------------------------------------



if __name__ == "__main__":
    main()