from typing import List, Tuple

BOS_TOKEN = "<BOS>"

KEEP = "KEEP"
DELETE = "DELETE"

def apply_tag(token: str, tag: str) -> List[str]:
    """
    Deterministically apply one edit tag to one token.
    """
    if tag == KEEP:
        return [token]

    if tag == DELETE:
        return []

    if tag.startswith("REPLACE_ADD_"):
        payload = tag[len("REPLACE_ADD_"):]
        repl, add = payload.split("|||", 1)
        out = [repl]
        if add.strip():
            out.extend(add.split("@@"))
        return out

    if tag.startswith("REPLACE_"):
        x = tag[len("REPLACE_"):]
        return [x]

    if tag.startswith("KEEP_ADD_"):
        y = tag[len("KEEP_ADD_"):]
        out = [token]
        if y.strip():
            out.extend(y.split("@@"))
        return out

    raise ValueError(f"Unknown tag: {tag}")


def apply_edit_tags(model_tokens: List[str], tags: List[str], bos_token: str = BOS_TOKEN) -> List[str]:
    """
    Apply edit tags to model tokens and reconstruct target-like sequence.
    """
    if len(model_tokens) != len(tags):
        raise ValueError("model_tokens and tags must have same length")

    output: List[str] = []
    for tok, tag in zip(model_tokens, tags):
        pieces = apply_tag(tok, tag)
        if tok == bos_token:
            # do not output BOS itself
            pieces = [p for p in pieces if p != bos_token]
        output.extend(pieces)
    return output

def pretty_print(src_line: str, tgt_line: str) -> None:
    model_tokens, tags = tree_pair_to_edit_tags(src_line, tgt_line)
    recovered = apply_edit_tags(model_tokens, tags)

    print("SOURCE:")
    print(src_line)
    print("\nTARGET:")
    print(tgt_line)
    print("\nMODEL TOKENS / TAGS:")
    for tok, tag in zip(model_tokens, tags):
        print(f"{tok:<15} -> {tag}")

    print("\nRECOVERED TARGET:")
    print(" ".join(recovered))


if __name__ == "__main__":
    examples = [
        (
            "(S PRON )NP (VP AUX VERB )VP )S",
            "(S PRON )NP (VP AUX (VP VERB )VP )VP )S",
        ),
        (
            "(S PRON )NP (VP AUX VERB )VP",
            "(S PRON )NP (VP AUX VERB )VP )S",
        ),
        (
            "(S PRON )NP (VP AUX VERB )VP )S",
            "(S PRON )NP (VP AUX VERB )S )S",
        ),
    ]

    for i, (src, tgt) in enumerate(examples, 1):
        print("=" * 80)
        print(f"EXAMPLE {i}")
        pretty_print(src, tgt)
        print()
        