from conllu import TokenList, parse_incr
from nltk.tree import Tree
from tqdm import tqdm

from steps.dep2const import tree2sentence

def mrg_to_conllu(lang, read_tree_path, read_conllu_path, write_path):
    illformed_count = 0
    total_count = 0
    sent_id = 1

    with open(write_path, "w", encoding="utf-8") as fout:
        pass

    with open(read_tree_path, "r", encoding="utf-8") as ftree, \
         open(read_conllu_path, "r", encoding="utf-8") as fconll:
        for tree, tokenlist in tqdm(zip(ftree, parse_incr(fconll)), desc="Converting trees to sentences"):
            tokens, is_illformed = tree2sentence(lang, Tree.fromstring(tree))
            conllu = TokenList(tokens, metadata=tokenlist.metadata).serialize()
            
            sent_id += 1 
            total_count += 1

            if is_illformed:
                illformed_count += 1

            with open(write_path, "a", encoding="utf-8") as fout:
                fout.write(conllu)
    print(f"Pecentage of illformed sentences: {illformed_count/total_count:.2%}")

def main():
    read_path = "predictions/en-penn-ud,filter=none,method=most-crossed,pos=upos,epoch=20.mrg"
    write_path = "predictions/en-penn-ud,filter=none,method=most-crossed,pos=upos,epoch=20conllu"
    mrg_to_conllu(read_path, write_path)

if __name__ == "__main__":
    main()



 

    


