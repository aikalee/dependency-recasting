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

    if pos_type.lower() not in ["upos", "xpos"]:
        print(f"Warning: Invalid POS tag '{pos_type}'. Using 'UPOS' by default.")
        pos_type = "UPOS"  # default to UPOS if not specified correctly

    dlookup = sentencedata.dlookup
    logger.debug(f"Head to Dependent Dictionary: {dlookup}")

   
    
    tree = get_subtree(dlookup[0][0], tokenlist, dlookup)

    if add_starting_node:
        return Tree("TOP", [tree])
    else:
        return tree
