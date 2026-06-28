from copy import deepcopy
from typing import List


def linearized_tree_to_structured_tokens(line: str, overlap: int = 0) -> List[dict]:
    """
    Convert linearized constituency tree into per-token features.

    Rules:
      - consecutive '(' before an anchor → that anchor's "left"
      - consecutive ')' after an anchor → that anchor's "right"
      - '(' after an anchor belong to next anchor (not current right)
    """
    seq = line.strip().split()
    n = len(seq)
    i = 0

    pending_left: List[str] = []
    results: List[dict] = []

    while i < n:
        tok = seq[i]

        # case 1: left brackets
        if tok.startswith("("):
            pending_left.append(tok)
            i += 1
            continue

        # case 2: invalid (right bracket before anchor)
        if tok.startswith(")"):
            raise ValueError(f"Unexpected right bracket {tok!r} before anchor at position {i}")

        # case 3: anchor token
        anchor = tok
        i += 1

        right: List[str] = []

        # collect only consecutive right brackets
        while i < n and seq[i].startswith(")"):
            right.append(seq[i])
            i += 1

        results.append(
            {
                "token": anchor,
                "left": pending_left.copy(),
                "right": right,
            }
        )

        # reset for next token
        pending_left = []
 
    if overlap > 0:
        orig_results = deepcopy(results)
        for i in range(len(results)):
            left_overlap = []
            right_overlap = []

            if i > 0:
                left_overlap = orig_results[i-1]["right"][-overlap:]
            
            if i + 1 < len(results):
                right_overlap = orig_results[i+1]["left"][:overlap]
            
            results[i]["left"] = left_overlap + results[i]["left"]
            results[i]["right"] = results[i]["right"] + right_overlap

    # sanity check
    if pending_left:
        raise ValueError(f"Dangling left brackets at end: {pending_left}")

    return results