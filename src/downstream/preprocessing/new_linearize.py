from nltk import Tree


def normalize_label(label: str) -> str:
    if label == ".":
        return "PUNCT"
    if "↓" in label:
        return label.replace("↓", "-down")
    if "↑" in label:
        return label.replace("↑", "-up")
    return label


def linearize_tree(tree_str: str) -> str:
    """
    Linearize bracketed tree into tokens like:
      (NP NN )NP

    Keep:
      - open labels:  (NP
      - close labels: )NP
      - POS tags:     NN

    Drop:
      - lexical words
    """
    tree = Tree.fromstring(tree_str)
    output = []

    def walk(node):
        # lexical leaf -> skip
        if isinstance(node, str):
            return

        label = normalize_label(node.label())

        # preterminal: (POS word)
        if len(node) == 1 and isinstance(node[0], str):
            output.append(label)
            return

        # internal constituent open
        output.append(f"({label}")

        # children in left-to-right order
        for child in node:
            walk(child)

        # internal constituent close
        output.append(f"){label}")

    walk(tree)
    return " ".join(output)