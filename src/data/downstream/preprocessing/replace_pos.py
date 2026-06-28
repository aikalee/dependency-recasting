from nltk import Tree
import copy

def get_sentence_pos(tree):
    """
    Get predicted POS from upstream predictions
    """
    sent_pos = []
    for child in tree:
        if isinstance(child[0], str):
            sent_pos.append(child.label())
        else:
            sent_pos.extend(get_sentence_pos(child))
    return sent_pos

def replace_sentence_pos(tree, sent_pos):
    """
    Replace sentence POS in gold constituents
    """
    tree
    for idx, pos in zip(tree.treepositions("leaves"), sent_pos, strict=True):
        tree[idx[:-1]] = Tree(pos, [tree[idx[:-1]][0]])

    return tree

def replace_pos_downstream_preprocessing(gold_const_file, pred_const_file, output_const_file):
    with open(output_const_file, "w", encoding="utf-8") as fout:
        pass
    with open(gold_const_file, encoding="utf-8") as fgold, \
         open(pred_const_file, encoding="utf-8") as fpred:
        for gold_ln, pred_ln in zip(fgold, fpred):
            pred_tree = Tree.fromstring(pred_ln)
            gold_tree = Tree.fromstring(gold_ln)
            sent_pos = get_sentence_pos(pred_tree)
            new_tree = replace_sentence_pos(gold_tree, sent_pos)
            with open(output_const_file, "a", encoding="utf-8") as fout:
                fout.write(new_tree.pformat(margin=100000) + "\n")

def main():
    """
    Input: (1) Predicted POS from upstream_outputs, (2) Gold Constituentized files
    Output: A new file with gold constituents and predicted POS 
    """
    
    s = "(TOP (root (nsubj (NOUN Jimbulang)) (det (DET aw)) (NOUN wayndare) (cop (AUX la)) (acl_relcl (nsubj (nmod (PRON wu)) (NOUN yitteem)) (aux (AUX doon)) (VERB ëmb) (obj (NOUN mbooleem) (nmod (NOUN xeeti) (nmod (NOUN xam-xam)))) (conj (cc (CCONJ walla)) (VERB man) (aux (AUX naa)) (xcomp (advmod (ADV itam)) (VERB tënku) (obl (case (ADP ci)) (det (DET benn)) (NOUN xam-xam))))) (appos (punct (PUNCT ,)) (cc (CCONJ maanaam)) (NOUN jimbulang) (acl_relcl (nsubj (PRON bu)) (aux (AUX di)) (VERB wax) (obl (case (ADP ci)) (NOUN paj) (conj (cc (CCONJ walla)) (case (ADP ci)) (NOUN taariix))) (advmod (ADV kepp)))) (punct (PUNCT .))))"
    tree = Tree.fromstring(s)
    sent_pos = get_sentence_pos(tree)
    new_tree = replace_sentence_pos(tree, sent_pos)
    print(new_tree)

if __name__ == "__main__":
    main()