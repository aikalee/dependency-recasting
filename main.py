import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from src.common.preprocessing.pipeline import common_preprocessing_pipeline
from src.common.postprocessing.pipeline import rulebased_postprocessing_pipeline
from src.downstream.preprocessing.pipeline import neural_preprocessing_pipeline
from src.downstream.postprocessing.pipeline import neural_postprocessing_pipeline

# === for debugging ===
from src.common.conllu_io import read_conllu
from src.common.preprocessing.dep2const import sentence2tree
from nltk import Tree




def main():
    
    neural_postprocessing_pipeline("Ancient_Greek", "UPOS", 100)
    # preprocessing_pipeline("Ancient_Greek", ["train", "dev", "test"], "UPOS", 100, is_target=False)
    # preprocessing_pipeline("Ancient_Greek", ["train", "dev", "test"], "XPOS")
    # postprocessing_pipeline("English", False, "UPOS", ["20", "100"])
    # preprocessing_pipeline("Chinese", ["train", "dev", "test"])
    # postprocessing_pipeline("Polish", "stanza", "finetune", "no", "yes", 100)
    # DATA = ROOT / "data" / "debug.conllu"
    # for tokenlist, sentencedata in read_conllu(DATA):
    #     tree = sentence2tree(sentencedata, tokenlist)
    # for conllu in read_conllu_file(DATA):
    #     tree = sentence2tree(conllu)
    #     print(tree)
   
    # for tokenlist, sentencedata in read_conllu(ROOT / "data" / "debug.conllu"):
    #         _ = sentence2tree(sentencedata, tokenlist)



if __name__ == "__main__":
    main()