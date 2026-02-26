import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from src.common.preprocessing.pipeline import common_preprocessing_pipeline
from src.downstream.preprocessing.pipeline import downstream_preprocessing_pipeline
from src.downstream.postprocessing.pipeline import postprocessing_pipeline

# === for debugging ===
from nltk.tree import Tree
from src.common.conllu_io import rewrite_conllu
from src.common.postprocessing.mrg_to_conllu import mrg_to_conllu
from src.downstream.postprocessing.txt2mrg import txt2mrg


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
    # common_preprocessing_pipeline(["Chinese"], ["train", "dev", "test"], "UPOS")
    # downstream_preprocessing_pipeline("Finnish", ["train", "dev", "test"], "UPOS", 100, is_target=True)
    postprocessing_pipeline("Finnish", "UPOS", 100, is_neural=False)
    # preprocessing_pipeline("Ancient_Greek", ["train", "dev", "test"], "UPOS", 100, is_target=False)
    # preprocessing_pipeline("Ancient_Greek", ["train", "dev", "test"], "XPOS")
    # postprocessing_pipeline("English", False, "UPOS", ["20", "100"])
    # preprocessing_pipeline("Chinese", ["train", "dev", "test"])
    # postprocessing_pipeline("Polish", "stanza", "finetune", "no", "yes", 100)
    # DATA = ROOT / "debug.txt"
    # txt2mrg(ROOT / "debug.txt", ROOT / "debug.mrg", ROOT / "debug.conllu", ROOT / "debug.output.mrg")
    # mrg_to_conllu("Finnish", ROOT / "debug.output.mrg", ROOT / "debug.conllu", ROOT / "debug.output.conllu")
    # rewrite_conllu(ROOT / "debug.output.conllu", ROOT / "debug.output.deprojz.conllu", False)
    # print(Tree.fromstring('(TOP (root (obj (NOUN Palon)) (VERB kerrotaan) (ccomp↓ (VERB saaneen) (obj (NOUN alkunsa)) (obl (acl (obl (nmod_poss (NOUN matkakeskuksen)) (NOUN pysäköintikerroksessa)) (VERB olleista)) (NOUN styroksilevyistä)) (conj (cc (CCONJ ja)) (VERB tuhonneen) (obj (amod (nmod_poss (nummod (advmod (ADV noin)) (NUM 5 000)) (NOUN neliömetrin)) (ADJ kokoisen)) (NOUN alueen)))) (punct (PUNCT .))))'))
    
    # for tokenlist, sentencedata in read_conllu(DATA):
    #     deprojz = deprojectivize_by_path(sentencedata)
    #     print(deprojz)


if __name__ == "__main__":
    main()