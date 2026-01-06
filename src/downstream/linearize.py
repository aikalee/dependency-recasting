def parse_tree(tree_str):
    tree_str += " "
    _, tree = _parse(tree_str, 0)
    return tree


def _parse(tree_str, i):
    assert tree_str[i] == '('
    i += 1                                                                                       # string pointer

    label = None
    children = []

    while tree_str[i] != ')':                                                                    # collect all the children (e.g., "(discourse" before seeing a ")"
        if tree_str[i] == '(':                                                                   # depth-first search, will return (None, chilren)
            i, child = _parse(tree_str, i)                                                       # children = [(label, children), (label, children)]
            children.append(child)
        else:
            if label is None:                                                                    # the terminal (None, children) trigger this condition
                r = min(p for p in (tree_str.find(' ', i), tree_str.find(')', i)) if p != -1)    # return ("POS", children) and will be appended to `children` of its parent
                label = tree_str[i:r]                                                        
                i = r                                                                   
            else:
                r = tree_str.find(')', i)                                                        # the end of a subtree is reached
                i = r

        if tree_str[i] == ' ':
            i += 1

    return i + 1, (label, children)                                                       

def flatten(tree):
    label, children = tree
    if children:
        return f"({label} " + " ".join(map(flatten, children)) + f" ){label}"
    else:
        return label

def linearize(tree_str):
    return flatten(parse_tree(tree_str))