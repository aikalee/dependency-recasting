import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from src.upstream.replace_bracket import replace_bracket_upstream_inference
from src.common.preprocessing.pipeline import common_preprocessing_pipeline
from src.downstream.preprocessing.pipeline import downstream_preprocessing_pipeline
from src.downstream.preprocessing.replace_pos import replace_pos_downstream_preprocessing
from src.downstream.preprocessing.mrg2txt import mrg2txt
from src.common.postprocessing.pipeline import postprocessing_pipeline

# === for debugging ===
from nltk.tree import Tree
from src.common.conllu_io import read_conllu, rewrite_conllu, reconstruct_conllu
from src.common.postprocessing.mrg_to_conllu import mrg_to_conllu
from src.downstream.postprocessing.txt2mrg import txt2mrg
from src.common.preprocessing.dep2const import sentence2tree
from src.common.postprocessing.deprojectivize import deprojectivize_by_path
from src.common.preprocessing.projectivize import projectivize, relabel


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
    # replace_bracket_upstream_inference("Ancient_Greek", ["train", "dev", "test"])
    common_preprocessing_pipeline(["Ancient_Greek"], ["train", "dev", "test"], "UPOS")
    downstream_preprocessing_pipeline("Ancient_Greek", ["train", "dev", "test"], "UPOS", 100, is_target=True)
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
        
    # txt2mrg(ROOT / "debug.txt", ROOT / "debug.mrg", ROOT / "debug.conllu", ROOT / "debug.output.mrg")
    # mrg_to_conllu("Finnish", ROOT / "debug.output.mrg", ROOT / "debug.conllu", ROOT / "debug.output.conllu")
    # rewrite_conllu(ROOT / "debug.output.conllu", ROOT / "debug.output.deprojz.conllu", False)
    # print(Tree.fromstring('(TOP (root (obj (NOUN Palon)) (VERB kerrotaan) (ccomp↓ (VERB saaneen) (obj (NOUN alkunsa)) (obl (acl (obl (nmod_poss (NOUN matkakeskuksen)) (NOUN pysäköintikerroksessa)) (VERB olleista)) (NOUN styroksilevyistä)) (conj (cc (CCONJ ja)) (VERB tuhonneen) (obj (amod (nmod_poss (nummod (advmod (ADV noin)) (NUM 5 000)) (NOUN neliömetrin)) (ADJ kokoisen)) (NOUN alueen)))) (punct (PUNCT .))))'))

    # for tokenlist, sentencedata in read_conllu(DATA):
    #     print(sentencedata.deprels)
    #     print(sentencedata.arcs)
    #     deprojz = sentence2tree(sentencedata, tokenlist)
    #     print(deprojz)


if __name__ == "__main__":
    main()