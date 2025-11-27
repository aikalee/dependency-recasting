import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
# from src.preprocessing.pipeline import preprocessing_pipeline
from src.postprocessing.pipeline import postprocessing_pipeline



def main():
    postprocessing_pipeline("English", False, "UPOS", ["20", "100"])
    # preprocessing_pipeline("English", ["train", "dev", "test"], [False, True], "UPOS")
    # DATA = ROOT / "data" / "debug.conllu"
    # for tokenlist, sentencedata in read_conllu(DATA):
    #     tree = sentence2tree(sentencedata, tokenlist)
    # for conllu in read_conllu_file(DATA):
    #     tree = sentence2tree(conllu)
    #     print(tree)
    # with open(DATA, "r", encoding="utf-8") as fin:
    #     for line in fin:
    #         tokenlist, is_illformed = tree2sentence(Tree.fromstring(line))
    #         print(TokenList(tokenlist).serialize())



if __name__ == "__main__":
    main()