import sys
from conllu import TokenList
from collections import defaultdict, deque
from functools import reduce
from operator import getitem

import logging
logger = logging.getLogger(__name__)
from nltk.tree import Tree

def sentence2tree(sentencedata, tokenlist, pos_type="UPOS", add_starting_node=True, replace_colon=True, replace_bracket=True):
    """
    Converts a sentence represented as a dictionary into a Tree object.
    Args:
        sentence (dict): The sentence data, where keys are token IDs and values are dictionaries with token attributes.
        pos_type (str): The part-of-speech tag to include in the tree nodes. Choose between "UPOS" or "XPOS". (set UPOS by default if the value is not in the options)

    """
    sent_id = 0
    def get_subtree(head_id, tokenlist, dlookup):
        """
        Recursively constructs a Tree object from the sentence data.
        Args:
            id (int): The ID of the head.
            sentence (dict): The sentence data, where keys are token IDs and values are dictionaries with token attributes.
            head2dep_dict (dict): A dictionary mapping heads to their dependent IDs.
        Returns:
            Tree: A Tree object representing the syntactic structure of the subtree below the head id.
        """
        
        children = dlookup.get(head_id, [])
        logger.debug(f"ID: {head_id}, Children: {children}")

        all_id = [head_id] + children
        all_id.sort()

        branches = []
        for current_id in all_id:
    
            logger.debug(f"Processing ID: {current_id}")
            if current_id == head_id:
                if pos_type == "UPOS":
                    pos = tokenlist[current_id-1]['upos']
                else:
                    pos = tokenlist[current_id-1]['xpos']
                if replace_colon:
                    pos = pos.replace(":", "_")
                form = tokenlist[current_id-1]['form']
                if replace_bracket:
                    form = form.replace("（", "-LRB-").replace("）", "-RRB-").replace("(", "-LRB-").replace(")", "-RRB-") # Penn-treebank standard
                branches.append(Tree(pos, [form]))
                    
            else:
                branches.append(get_subtree(current_id, tokenlist, dlookup))
        deprel = tokenlist[head_id-1]['deprel']
        if replace_colon:
            deprel = deprel.replace(":", "_")
        return Tree(deprel, branches)

    if pos_type not in ["UPOS", "XPOS"]:
        print(f"Warning: Invalid POS tag '{pos_type}'. Using 'UPOS' by default.")
        pos_type = "UPOS"  # default to UPOS if not specified correctly

    dlookup = sentencedata.dlookup
    logger.debug(f"Head to Dependent Dictionary: {dlookup}")

   
    
    tree = get_subtree(dlookup[0][0], tokenlist, dlookup)
    
    if add_starting_node:
        return Tree("TOP", [tree])
    else:
        return tree

def add_ids_to_tree(tree):
    """
    Add sequential IDs to leaf nodes of an NLTK Tree.
    Each word becomes a tuple: (ID, word), starting from 1.
    Returns a new tree with IDs added.
    """
    counter = [1]  # mutable counter to track word position

    def _add_ids(t):
        logger.debug(f"Processing node: {t}")
        if isinstance(t, (str, tuple)):
            # Replace word with (id, word)
            word_id = counter[0]
            counter[0] += 1
            if isinstance(t, str):
                return (str(word_id), t)
            else:
                return (str(word_id),) + t

        elif isinstance(t, Tree):
            return Tree(t.label(), [_add_ids(child) for child in t])
        else:
            return t  # already tagged or invalid

    return _add_ids(tree)

def check_illformed(subtree):
    heads = [child for child in subtree if isinstance(child[0], str)]
    if len(heads) != 1:
        return True
    for child in subtree:
        if not isinstance(child[0], str):
            if check_illformed(child):
                return True
    return False

def fix_illformed(tree):

    def _assign_parents(tree, parent=None):
        """
        Recursively assign _parent and _index attributes to each node.
        This allows bottom-up updates without external tree maps.
        """
        tree._parent = parent
        for i, child in enumerate(tree):
            if isinstance(child, Tree):
                child._index = i
                _assign_parents(child, tree)

    def _build_tree(original_child, updated_child):
        """
        Rebuild tree upward by replacing original_child with updated_child.
        Uses parent pointers instead of id() maps.
        """
        parent = getattr(original_child, "_parent", None)

        # reached root
        
        if parent is None or parent.label() == "TOP":
            return updated_child
     
        index = getattr(original_child, "_index", None)
        if index is None:
            raise ValueError("Child index not found.")

        # rebuild parent and propagate upward
        updated_parent = parent.copy()
        updated_parent.remove(original_child)
        updated_parent.insert(index, updated_child)
        _assign_parents(updated_parent, parent._parent)  # reassign links
        return _build_tree(parent, updated_parent)
    
    def _find_shallowest_leaf(tree):
        """Return the shallowest (nearest) leaf-containing subtree."""
        queue = deque([(tree, 0)])  # (node, depth)
        while queue:
            branch, depth = queue.popleft()
            if isinstance(branch, Tree):
                # if its child is a terminal, this is a preterminal
                if any(isinstance(c, str) for c in branch):
                    return branch, depth + 1
                for child in branch:
                    queue.append((child, depth + 1))
        return None, None
    
    def _find_siblings(head_chosen, depth):
        node = head_chosen
        layers_2b_unwrapped = depth-1

        for _ in range(layers_2b_unwrapped-1):
            prev_node = node
            node = prev_node[0]

        return list(node)
    
    def _select_head(subtree):
        heads = []
        head_chosen = None
        wrong_heads_2b_lowered = []

        heads = [child for child in subtree if isinstance(child, Tree) and isinstance(child[0], str)] # added isinstance(child, Tree) to handle edge cases
        
        if len(heads) > 1:
            head_chosen = heads[-2] if heads[-1].label() == "punct" else heads[-1]
            wrong_heads_2b_lowered = [head for head in heads if id(head) != id(head_chosen)]  # use id to refer the unique object
        elif len(heads) == 0:
            head_chosen = subtree[-2] if subtree[-1].label() == "punct" else subtree[-1]

        return head_chosen, wrong_heads_2b_lowered
        
    def _lift_chosen_head(head_chosen, subtree):
        _, depth = _find_shallowest_leaf(head_chosen)
        unwrapped_head_chosen = _find_siblings(head_chosen, depth)
        head_chosen_index = head_chosen._index
        updated_subtree = subtree.copy()
        updated_subtree.remove(head_chosen)
        for offset, child in enumerate(unwrapped_head_chosen):
            updated_subtree.insert(head_chosen_index+offset, child)
        return updated_subtree
        
    _assign_parents(tree)
    queue = deque([tree]) # `tree` only refer to the entire tree here
      
    while queue:

        updated_subtree = None # reset updated subtree
        subtree = queue.popleft()
        
        head_chosen, wrong_heads_2b_lowered = _select_head(subtree)

        # update current tree
        # step 1: record the original positions
        # step 2: remove the subtree
        # step 3: attach the branches back
        # step 4: reset 
        if head_chosen:
            if wrong_heads_2b_lowered:
                for wrong_head in wrong_heads_2b_lowered:
                    tree = _build_tree(wrong_head, Tree("dummy", [wrong_head]))
                    _assign_parents(tree)
            else:
                updated_subtree = _lift_chosen_head(head_chosen, subtree)
                head_chosen, wrong_heads_2b_lowered = _select_head(updated_subtree) 
               
                if wrong_heads_2b_lowered:
                    for wrong_head in wrong_heads_2b_lowered:
                        idx = updated_subtree.index(wrong_head)
                        updated_subtree.remove(wrong_head)
                        updated_subtree.insert(idx, Tree("dummy", [wrong_head]))
                tree = _build_tree(subtree, updated_subtree)
                _assign_parents(tree)
            head_chosen = None
            wrong_heads_2b_lowered = []
       
        subtree_for_queuing = updated_subtree if updated_subtree else subtree
        for child in list(subtree_for_queuing):
            if not isinstance(child[0], str):
                queue.append(child)
    return tree

def tree2sentence(lang, tree, pos_type="UPOS"):
    """
    Recursively extracts from a Tree object (from sentence2tree) and returns the sentence as a dictionary of words
        dictionary format: {id: {"form": word, "head": head_id, "deprel": dep_rel}}
        id starts from 1, and the root has id 0.
    """
        
    def find_head_id(subtree):
        """
        Find the head ID of the subtree.
        The head of the token which is not in a tree format.
        """
        for child in subtree:
            if len(child) == 1 and isinstance(child[0], tuple):
                return child[0][0]
            
    def add_head_to_sentence(subtree, tokens, head_id=0, deprel="ROOT"):
        """
        Add the head ID and dependency relation to the sentence dictionary.
        """
        logger.debug(f"Processing subtree: {subtree}, head_id: {head_id}")
        for child in subtree: 
            logger.debug(f"Processing child: {child}")
            if isinstance(child, tuple):
                upos = subtree.label() if pos_type == "UPOS" else None
                xpos = subtree.label() if pos_type == "XPOS" else None
                deprel = deprel.replace("_", ":")
                form = child[1].replace("-LRB-", "（").replace("-RRB-", "）") if lang == "Chinese" else child[1].replace("-LRB-", "(").replace("-RRB-", ")")
                logger.debug(f"Adding token: {child}, head: {head_id}, {pos_type}: {subtree.label()}, deprel: {deprel}")
                token = {
                    "id": int(child[0]),
                    "form": form,
                    "lemma": None,
                    "upos": upos,
                    "xpos": xpos,
                    "feats": None,
                    "head": int(head_id),
                    "deprel": deprel,
                    "deps": None,
                    "misc": None
                }
                tokens.insert(int(child[0]), token)
                logger.debug(f"Current sentence state: {tokens}")    
            elif len(child) == 1 and isinstance(child[0], tuple): 
                add_head_to_sentence(child, tokens, head_id, subtree.label())
            else:
                subtree_head_id = find_head_id(subtree) # not all subtrees have children
                add_head_to_sentence(child, tokens, subtree_head_id, subtree.label())

    if pos_type not in ["UPOS", "XPOS"]:
        print(f"Warning: Invalid POS tag '{pos_type}'. Using 'UPOS' by default.")
        pos_type = "UPOS"  # default to UPOS if not specified correctly

    #remove starting node if it exists
    if len(tree) == 1:
        tree = tree[0]
    
    is_illformed = check_illformed(tree)
    if is_illformed:
        tree = fix_illformed(tree)
        
    tree_with_ids = add_ids_to_tree(tree)
    logger.info(f"Tree with IDs: {tree_with_ids}")
    tokens = []
    add_head_to_sentence(tree_with_ids, tokens, 0)
    return tokens, is_illformed

def main():
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)
    # for sentence, sent_id in  read_conllu_file("draft_sentences.conllu", return_sent_ids=True):
    #     if sent_id != "trial-2":
    #         continue
    #     print(sentence)
    sentence = {
        1: {'id': 1, 'form': 'the', 'lemma': 'the', 'UPOS': 'DET', 'XPOS': 'DT', 'feats': 'Definite=Def|PronType=Art', 'head': 3, 'deprel': 'det', 'deps': '_', 'misc': '_'},
        2: {'id': 2, 'form': 'hungry', 'lemma': 'hungry', 'UPOS': 'ADJ', 'XPOS': 'JJ', 'feats': 'Degree=Pos', 'head': 3, 'deprel': 'amod', 'deps': '_', 'misc': '_'},
        3: {'id': 3, 'form': 'cat', 'lemma': 'cat', 'UPOS': 'NOUN', 'XPOS': 'NN', 'feats': 'Number=Sing', 'head': 4, 'deprel': 'nsubj', 'deps': '_', 'misc': '_'},
        4: {'id': 4, 'form': 'ate', 'lemma': 'eat', 'UPOS': 'VERB', 'XPOS': 'VBD', 'feats': 'Tense=Past|VerbForm=Fin', 'head': 0, 'deprel': 'root', 'deps': '_', 'misc': '_'},
        5: {'id': 5, 'form': 'a', 'lemma': 'a', 'UPOS': 'DET', 'XPOS': 'DT', 'feats': 'Definite=Ind|PronType=Art', 'head': 7, 'deprel': 'det', 'deps': '_', 'misc': '_'},
        6: {'id': 6, 'form': 'red', 'lemma': 'red', 'UPOS': 'ADJ', 'XPOS': 'JJ', 'feats': 'Degree=Pos', 'head': 7, 'deprel': 'amod', 'deps': '_', 'misc': '_'},
        7: {'id': 7, 'form': 'apple', 'lemma': 'apple', 'UPOS': 'NOUN', 'XPOS': 'NN', 'feats': 'Number=Sing', 'head': 4, 'deprel': 'obj', 'deps': '_', 'misc': '_'}}
    
    tree = sentence2tree(sentence, pos_type="XPOS")
    print(tree)
    tree.pretty_print()

    sentence_new = tree2sentence(tree, pos_type="XPOS")
    for token in sentence_new.items():
        print(token)

    #check data type
    print(f"Type of tree: {type(tree)}")

def test():
    logging.basicConfig(level=logging.ERROR)
    
    file_path = "../../data/UD_Chinese-GSD/zh_gsd-ud-train.conllu"
    count_total = 0
    count_nonprojective = 0
    count_wrong = 0
    count_wrong_non_projective = 0
    
    for sentence, sent_id in read_conllu_file(file_path, return_sent_ids=True):
        # keep only id, form, UPOS, head, deprel
        sentence = {k: {"id": v['id'], "form": v['form'], "UPOS": v['UPOS'], "head": v['head'], "deprel": v['deprel']} for k, v in sentence.items()}
        tree = sentence2tree(sentence)
        sentence_new = tree2sentence(tree)
        
        count_total += 1
        if is_non_projective_sentence(sentence):
            count_nonprojective += 1
        
        if sentence_new != sentence:
            logger.error(f"Sentence {sent_id} is not equal after conversion.")
            logger.error(f"Original: {sentence}")
            logger.error(f"Converted: {sentence_new}")
            count_wrong += 1
            if is_non_projective_sentence(sentence):
                count_wrong_non_projective += 1

        
    print()
    print(f"Total sentences: {count_total}, Wrong sentences: {count_wrong}")
    print(f"Total non-projective sentences: {count_nonprojective}, Wrong non-projective sentences: {count_wrong_non_projective}")
    print()
    print(f"Error rate: {count_wrong / count_total:.2%}")
    print(f"Error rate for non-projective sentences: {count_wrong_non_projective / count_total:.2%}")


    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        main()
    elif sys.argv[1] == "test":
        test()