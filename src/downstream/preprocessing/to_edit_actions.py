from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Tuple


BOS_TOKEN = "<BOS>"

KEEP = "KEEP"
DELETE = "DELETE"


def is_structure_token(tok: str) -> bool:
    return tok.startswith("(") or tok.startswith(")")


def is_anchor_token(tok: str) -> bool:
    return not is_structure_token(tok)


def validate_anchor_sequence(src_tokens: List[str], tgt_tokens: List[str]) -> None:
    src_anchors = [t for t in src_tokens if is_anchor_token(t)]
    tgt_anchors = [t for t in tgt_tokens if is_anchor_token(t)]

    if src_anchors != tgt_anchors:
        raise ValueError(
            "Anchor sequences do not match.\n"
            f"src anchors: {src_anchors}\n"
            f"tgt anchors: {tgt_anchors}"
        )


def split_by_anchors(tokens: List[str]) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """
    Split token sequence into:
      - prefix structure tokens before first anchor
      - list of (anchor, following_structure_segment)

    Example:
      ["(S", "PRON", "(VP", "AUX", ")VP", ")S"]
    =>
      prefix = ["(S"]
      pairs = [
        ("PRON", ["(VP"]),
        ("AUX", [")VP", ")S"])
      ]
    """
    prefix: List[str] = []
    pairs: List[Tuple[str, List[str]]] = []

    i = 0
    n = len(tokens)

    while i < n and is_structure_token(tokens[i]):
        prefix.append(tokens[i])
        i += 1

    while i < n:
        if not is_anchor_token(tokens[i]):
            raise ValueError(f"Expected anchor token at position {i}, got {tokens[i]!r}")

        anchor = tokens[i]
        i += 1

        seg: List[str] = []
        while i < n and is_structure_token(tokens[i]):
            seg.append(tokens[i])
            i += 1

        pairs.append((anchor, seg))

    return prefix, pairs


def attach_add(tags: List[str], idx: int, inserted_tokens: List[str]) -> None:
    """
    Attach insertion to token idx.

    Supported transitions:
      KEEP -> KEEP_ADD_Y
      REPLACE_X -> REPLACE_ADD_X|||Y

    We intentionally keep label space small and reject more complex states.
    """
    if not inserted_tokens:
        return

    add_str = " ".join(inserted_tokens)
    current = tags[idx]

    if current == KEEP:
        tags[idx] = f"KEEP_ADD_{add_str}"
        return

    if current.startswith("REPLACE_") and not current.startswith("REPLACE_ADD_"):
        repl = current[len("REPLACE_"):]
        tags[idx] = f"REPLACE_ADD_{repl}|||{add_str}"
        return

    raise ValueError(
        f"Cannot attach ADD to existing tag {current!r}. "
        "This sample needs a richer edit schema."
    )


def segment_to_actions(src_seg: List[str], tgt_seg: List[str]) -> Tuple[List[str], List[str]]:
    """
    Convert one structure segment into token-aligned actions.

    Returns:
      actions_for_src_seg: same length as src_seg
      prefix_additions: tokens inserted before the whole segment
                        (to attach to BOS / previous anchor)
    """
    actions = [KEEP for _ in src_seg]
    prefix_additions: List[str] = []

    sm = SequenceMatcher(a=src_seg, b=tgt_seg, autojunk=False)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        elif tag == "delete":
            for k in range(i1, i2):
                actions[k] = DELETE

        elif tag == "insert":
            inserted = tgt_seg[j1:j2]
            if i1 == 0:
                prefix_additions.extend(inserted)
            else:
                attach_add(actions, i1 - 1, inserted)

        elif tag == "replace":
            src_chunk = src_seg[i1:i2]
            tgt_chunk = tgt_seg[j1:j2]

            src_len = len(src_chunk)
            tgt_len = len(tgt_chunk)
            common = min(src_len, tgt_len)

            # aligned replacements
            for off in range(common):
                actions[i1 + off] = f"REPLACE_{tgt_chunk[off]}"

            # extra source tokens -> delete
            for k in range(i1 + common, i2):
                actions[k] = DELETE

            # extra target tokens -> attach as ADD
            extra_insert = tgt_chunk[common:]
            if extra_insert:
                if i2 == 0:
                    prefix_additions.extend(extra_insert)
                else:
                    attach_add(actions, i2 - 1, extra_insert)

        else:
            raise RuntimeError(f"Unexpected opcode: {tag}")

    return actions, prefix_additions


def tree_pair_to_edit_tags(src_line: str, tgt_line: str, bos_token: str = BOS_TOKEN,) -> Tuple[List[str], List[str]]:
    """
    Convert one source-target tree pair into:
      model_tokens: [BOS] + source tokens
      tags: same length as model_tokens

    Tag semantics:
      KEEP                  -> output original token
      DELETE                -> output nothing
      REPLACE_X             -> output X
      KEEP_ADD_Y            -> output original token, then Y
      REPLACE_ADD_X|||Y     -> output X, then Y
    """
    src_tokens = src_line.strip().split()
    tgt_tokens = tgt_line.strip().split()

    validate_anchor_sequence(src_tokens, tgt_tokens)

    src_prefix, src_pairs = split_by_anchors(src_tokens)
    tgt_prefix, tgt_pairs = split_by_anchors(tgt_tokens)

    if len(src_pairs) != len(tgt_pairs):
        raise ValueError(f"Anchor pair count mismatch after validation. Length of source pairs {len(src_pairs)}, length of target pairs {len(tgt_pairs)}")

    model_tokens: List[str] = [bos_token]
    tags: List[str] = [KEEP]

    # Handle prefix structure tokens before first anchor
    prefix_actions, bos_adds = segment_to_actions(src_prefix, tgt_prefix)
    if bos_adds:
        tags[0] = f"KEEP_ADD_{' '.join(bos_adds)}"

    model_tokens.extend(src_prefix)
    tags.extend(prefix_actions)

    # Handle anchored segments
    for (src_anchor, src_seg), (tgt_anchor, tgt_seg) in zip(src_pairs, tgt_pairs):
        if src_anchor != tgt_anchor:
            raise ValueError(f"Anchor mismatch: {src_anchor!r} vs {tgt_anchor!r}")

        # Anchor token itself: always KEEP
        model_tokens.append(src_anchor)
        tags.append(KEEP)

        seg_actions, seg_prefix_adds = segment_to_actions(src_seg, tgt_seg)

        # Tokens inserted before this segment are attached to the anchor
        if seg_prefix_adds:
            attach_add(tags, len(tags) - 1, seg_prefix_adds)

        model_tokens.extend(src_seg)
        tags.extend(seg_actions)

    assert len(model_tokens) == len(tags)
    return model_tokens, tags


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
            out.extend(add.split())
        return out

    if tag.startswith("REPLACE_"):
        x = tag[len("REPLACE_"):]
        return [x]

    if tag.startswith("KEEP_ADD_"):
        y = tag[len("KEEP_ADD_"):]
        out = [token]
        if y.strip():
            out.extend(y.split())
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
        