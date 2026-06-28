from typing import List, Tuple
from nltk import Tree

def linearized_to_ptb(line: str, word_prefix: str = "word", start_index: int = 0) -> str:
    """
    Convert tokens like: (ROOT (VP VERB )VP )ROOT
    into PTB:          (ROOT (VP (VERB word0)))
    
    Rules:
      - token "(X"  => open constituent X
      - token ")X"  => close constituent X (we don't actually need X to print PTB)
      - other token => leaf tag T -> (T wordN)
    Returns: (ptb_string, next_word_index)
    """
    tokens = [t for t in line.strip().split() if t]
    out: List[str] = []
    w = start_index

    for tok in tokens:
        if tok.startswith("(") and len(tok) > 1:
            label = tok[1:]
            if "-down" in label:
                label = label.replace("-down", "↓")
            elif "-up" in label:
                label = label.replace("-up", "↑")
            out.append(f"({label}")
        elif tok.startswith(")") and len(tok) > 1:
            out.append(")")
        else:
            # leaf/preterminal tag
            out.append(f"({tok} {word_prefix}{w})")
            w += 1

    # PTB spacing cleanup (optional but nice)
    ptb = " ".join(out)
    ptb = ptb.replace("( ", "(").replace(" )", ")")
    return ptb

from typing import Tuple, List

def validate_linearized_brackets(line: str) -> Tuple[bool, str]:
    """
    Validates tokenized brackets format:
      (X opens X
      )X closes X and must match last open
    
    Returns (ok, message). If not ok, message explains the first error.
    """
    tokens = [t for t in line.strip().split() if t]
    stack: List[str] = []

    for i, tok in enumerate(tokens):
        if tok.startswith("(") and len(tok) > 1:
            lab = tok[1:]
            stack.append(lab)

        elif tok.startswith(")") and len(tok) > 1:
            lab = tok[1:]
            if not stack:
                return False, f"Extra closing {tok} at token {i}"
            top = stack[-1]
            if lab != top:
                return False, f"Mismatch at token {i}: got {tok} but expected ){top}"
            stack.pop()

        else:
            # normal label/terminal token, ignore for structure validation
            pass

    if stack:
        return False, f"Unclosed opens at end: {stack[-10:]} (showing last 10)"
    return True, "OK"



def main():
    s = "(TOP (root (ccomp (nsubj (compound PROPN )compound PROPN (cc CONJ )cc (conj (compound PROPN )compound NOUN )conj )nsubj (punct PUNCT )punct (aux AUX )aux (neg PART )neg VERB (xcomp (punct PUNCT )punct (mark PART )mark (cop VERB )cop (amod (advmod ADV )advmod ADJ )amod (det DET )det NOUN (advmod ADP (nmod (case ADP )case NUM )nmod )advmod )xcomp )ccomp (punct PUNCT )punct (nsubj (det DET )det NOUN )nsubj VERB (punct PUNCT )punct )root )TOP"
    new_s = linearized_to_ptb(s)
    t = Tree.fromstring(new_s)
    print(new_s)

if __name__ == "__main__":
    main()
