import logging
logger = logging.getLogger(__name__)

from collections import deque
from nltk.tree import Tree

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
        """Return the shallowest leaf-containing subtree."""
        queue = deque([(tree, 0)])                                                                      # (node, depth)
        while queue:
            branch, depth = queue.popleft()
            if isinstance(branch, Tree):
                if any(isinstance(c, str) for c in branch):                         
                    return branch, depth + 1
                for child in branch:
                    queue.append((child, depth + 1))
                    
        return None, None
    
    def _unwrap_chosen_branch(tree, depth, children, layers_unwrapped):

        if layers_unwrapped == depth-1:
            children.append(tree)
            return children
        else:
            layers_unwrapped += 1
            for t in tree:
                _unwrap_chosen_branch(t, depth, children, layers_unwrapped)
        return children
    
    def _select_head(subtree):
        heads = []
        head_chosen = None
        wrong_heads_2b_lowered = []

        heads = [child for child in subtree if isinstance(child, Tree) and isinstance(child[0], str)] # added isinstance(child, Tree) to handle edge cases
        
        if len(heads) > 1:
            head_chosen = heads[-2] if heads[-1].label() == "punct" else heads[-1]
            wrong_heads_2b_lowered = [head for head in heads if id(head) != id(head_chosen)]          # use id to refer the unique object
        elif len(heads) == 0:
            head_chosen = subtree[-2] if subtree[-1].label() == "punct" else subtree[-1]

        return head_chosen, wrong_heads_2b_lowered
        
    def _lift_chosen_head(chosen_branch, subtree):

        _, depth = _find_shallowest_leaf(chosen_branch)
        unwrapped_chosen_branch = _unwrap_chosen_branch(chosen_branch, depth, [], 0)
        branch_idx = chosen_branch._index

        updated_subtree = subtree.copy()
        updated_subtree.remove(chosen_branch)
        for offset, child in enumerate(unwrapped_chosen_branch):
            updated_subtree.insert(branch_idx+offset, child)
        return updated_subtree
        
    _assign_parents(tree)
    queue = deque([tree]) # `tree` only refer to the entire tree here
   
    while queue:
    # for i in range(2):

        updated_subtree = None # reset updated subtree
        subtree = queue.popleft()
    
        chosen_branch, wrong_heads_2b_lowered = _select_head(subtree)
        

        # update current tree
        # step 1: record the original positions
        # step 2: remove the subtree
        # step 3: attach the branches back
        # step 4: reset 
        if chosen_branch:
            # Case 1: Mutiple heads
            if wrong_heads_2b_lowered:
                for wrong_head in wrong_heads_2b_lowered:
                    tree = _build_tree(wrong_head, Tree("dummy", [wrong_head]))
                    _assign_parents(tree)
            # Case 2: No heeads
            else:
                updated_subtree = _lift_chosen_head(chosen_branch, subtree)
                chosen_branch, wrong_heads_2b_lowered = _select_head(updated_subtree) 
                if wrong_heads_2b_lowered:
                    for wrong_head in wrong_heads_2b_lowered:
                        idx = updated_subtree.index(wrong_head)
                        updated_subtree.remove(wrong_head)
                        updated_subtree.insert(idx, Tree("dummy", [wrong_head]))
                tree = _build_tree(subtree, updated_subtree)
                _assign_parents(tree)
            chosen_branch = None
            wrong_heads_2b_lowered = []
       
        subtree_for_queuing = updated_subtree if updated_subtree else subtree
       
        for child in list(subtree_for_queuing):
            if not isinstance(child[0], str):
                queue.append(child)
    # print(tree)
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
                # upos = subtree.label() if pos_type == "UPOS" else None
                # xpos = subtree.label() if pos_type == "XPOS" else None
                deprel = deprel.replace("_", ":")
                # form = child[1].replace("-LRB-", "（").replace("-RRB-", "）") if lang == "Chinese" else child[1].replace("-LRB-", "(").replace("-RRB-", ")")
                logger.debug(f"Adding token: {child}, head: {head_id}, {pos_type}: {subtree.label()}, deprel: {deprel}")
                # token = {
                #     "id": int(child[0]),
                #     "form": child[1],
                #     "lemma": None,
                #     "upos": upos,
                #     "xpos": xpos,
                #     "feats": None,
                #     "head": int(head_id),
                #     "deprel": deprel,
                #     "deps": None,
                #     "misc": None
                # }
                # tokens.insert(int(child[0]), token)
                token = {"id": int(child[0]), "form": child[1], "head": head_id, "deprel": deprel}
                tokens.insert(int(child[0]), token)
                logger.debug(f"Current sentence state: {tokens}")    
            elif len(child) == 1 and isinstance(child[0], tuple): 
                add_head_to_sentence(child, tokens, head_id, subtree.label())
            else:
                subtree_head_id = find_head_id(subtree)                                               # not all subtrees have children
                add_head_to_sentence(child, tokens, subtree_head_id, subtree.label())

    if pos_type.lower() not in ["upos", "xpos"]:
        print(f"Warning: Invalid POS tag '{pos_type}'. Using 'UPOS' by default.")
        pos_type = "UPOS"                                                                             # default to UPOS if not specified correctly

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
    """
    Should be 
    (obj
        (nmod (ADJ τοίαν)) (NOUN Πέλοψ))
        (nmod
            (NOUN ἴυγγα)
            (nmod (ADJ θηρατηρίαν))
            (nmod (NOUN ἔρωτος))))
        (punct (PUNCT ,))
    """
    s = "(TOP (root (dep (nummod (compound (NUM 8)) (compound (NUM 13/16))) (dep (SYM %)) (dep (ADP to)) (compound (NUM 8)) (compound (NUM 11/16)) (SYM %) (nmod_npmod (nummod (NUM one)) (NOUN month)) (punct (PUNCT ;)) (dep (compound (NUM 8)) (compound (NUM 13/16)) (dep (SYM %)) (dep (ADP to)) (compound (NUM 8)) (compound (NUM 11/16)) (SYM %) (nmod_npmod (nummod (NUM two)) (NOUN months))) (punct (PUNCT ;)) (dep (compound (NUM 8)) (compound (NUM 13/16)) (dep (SYM %)) (dep (ADP to)) (compound (NUM 8)) (compound (NUM 11/16)) (SYM %) (nmod_npmod (nummod (NUM three)) (NOUN months))) (punct (PUNCT ;)) (dep (compound (NUM 8)) (compound (NUM 3/4)) (dep (SYM %)) (dep (ADP to)) (compound (NUM 8)) (compound (NUM 5/8)) (SYM %) (nmod_npmod (nummod (NUM four)) (NOUN months))) (punct (PUNCT ;)) (dep (compound (NUM 8)) (compound (NUM 11/16)) (dep (SYM %)) (dep (ADP to)) (compound (NUM 8)) (compound (NUM 9/16)) (SYM %) (nmod_npmod (nummod (NUM five)) (NOUN months))) (punct (PUNCT ;)) (dep (compound (NUM 8)) (compound (NUM 5/8)) (dep (SYM %)) (dep (ADP to)) (compound (NUM 8)) (compound (NUM 1/2)) (SYM %) (nmod_npmod (nummod (NUM six)) (NOUN months)))) (punct (PUNCT .))))"
    # s = "(TOP (root (obj (nmod (nmod (nmod (ADJ τοίαν)) (NOUN Πέλοψ)) (nmod (NOUN ἴυγγα) (nmod (ADJ θηρατηρίαν)) (nmod (NOUN ἔρωτος)))) (punct (PUNCT ,))) (obj↓ (NOUN ἀστραπὴν) (nmod (PRON τιν̓))) (obj (NOUN ὀμμάτων)) (VERB ἔχει) (punct (PUNCT ·))))"
    d = tree2sentence("English", Tree.fromstring(s))

if __name__ == "__main__":
    main()