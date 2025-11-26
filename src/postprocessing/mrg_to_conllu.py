from nltk.tree import Tree
from tqdm import tqdm

from steps.data_loader import sentence_to_conllu
from steps.dependency2constituency import tree2sentence

def mrg_to_conllu(read_path, write_path):
    illformed_count = 0
    total_count = 0
    sent_id = 1

    with open(write_path, "w", encoding="utf-8") as fout:
        pass

    with open(read_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Converting trees to sentences"):
            sentence, is_illformed = tree2sentence(Tree.fromstring(line))
            conllu = sentence_to_conllu(sentence, sent_id)
             
            sent_id += 1 
            total_count += 1

            if is_illformed:
                illformed_count += 1

            with open(write_path, "a", encoding="utf-8") as fout:
                fout.write(conllu + "\n")
    print(f"Pecentage of illformed sentences: {illformed_count/total_count:.2%}")

def main():
    read_path = "predictions/en-penn-ud,filter=none,method=most-crossed,pos=upos,epoch=20.mrg"
    write_path = "predictions/en-penn-ud,filter=none,method=most-crossed,pos=upos,epoch=20conllu"
    mrg_to_conllu(read_path, write_path)

if __name__ == "__main__":
    main()



 

    


