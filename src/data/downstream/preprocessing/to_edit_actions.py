from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Tuple, Optional


BOS_TOKEN = "<BOS>"

KEEP = "KEEP"
DELETE = "DELETE"
REPLACE = "REPLACE"
KEEP_ADD = "KEEP_ADD"
REPLACE_ADD = "REPLACE_ADD"


def is_structure_token(tok: str) -> bool:
    return tok.startswith("(") or tok.startswith(")") or tok == BOS_TOKEN


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


@dataclass
class MultiHeadTag:
    op: str
    replace: Optional[str]
    add: List[str]


def new_keep_tag() -> MultiHeadTag:
    return MultiHeadTag(op=KEEP, replace=None, add=[])


def new_delete_tag() -> MultiHeadTag:
    return MultiHeadTag(op=DELETE, replace=None, add=[])


def new_replace_tag(repl: str) -> MultiHeadTag:
    return MultiHeadTag(op=REPLACE, replace=repl, add=[])


def attach_add(tags: List[MultiHeadTag], idx: int, inserted_tokens: List[str]) -> None:
    """
    Attach insertion to token idx.

    Supported transitions:
      KEEP -> KEEP_ADD
      REPLACE -> REPLACE_ADD
    """
    if not inserted_tokens:
        return

    current = tags[idx]

    if current.op == KEEP:
        current.op = KEEP_ADD
        current.add.extend(inserted_tokens)
        return

    if current.op == REPLACE:
        current.op = REPLACE_ADD
        current.add.extend(inserted_tokens)
        return

    raise ValueError(
        f"Cannot attach ADD to existing op {current.op!r}. "
        "This sample needs a richer edit schema."
    )


def segment_to_actions(src_seg: List[str], tgt_seg: List[str]) -> Tuple[List[MultiHeadTag], List[str]]:
    """
    Convert one structure segment into token-aligned actions.

    Returns:
      actions_for_src_seg: same length as src_seg
      prefix_additions: tokens inserted before the whole segment
                        (to attach to BOS / previous anchor)
    """
    actions = [new_keep_tag() for _ in src_seg]
    prefix_additions: List[str] = []

    sm = SequenceMatcher(a=src_seg, b=tgt_seg, autojunk=False)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        elif tag == "delete":
            for k in range(i1, i2):
                actions[k] = new_delete_tag()

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
                actions[i1 + off] = new_replace_tag(tgt_chunk[off])

            # extra source tokens -> delete
            for k in range(i1 + common, i2):
                actions[k] = new_delete_tag()

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


def tree_pair_to_edit_tags(src_line: str, tgt_line: str, bos_token: str = BOS_TOKEN) -> dict:
    """
    Convert one source-target tree pair into a multi-head dictionary:

      {
        "tokens": [...],
        "op": [...],
        "replace": [...],
        "add": [...]
      }

    Semantics:
      KEEP         -> output original token
      DELETE       -> output nothing
      REPLACE      -> output `replace`
      KEEP_ADD     -> output original token, then everything in `add`
      REPLACE_ADD  -> output `replace`, then everything in `add`
    """
    src_tokens = src_line.strip().split()
    tgt_tokens = tgt_line.strip().split()

    validate_anchor_sequence(src_tokens, tgt_tokens)

    src_prefix, src_pairs = split_by_anchors(src_tokens)
    tgt_prefix, tgt_pairs = split_by_anchors(tgt_tokens)

    if len(src_pairs) != len(tgt_pairs):
        raise ValueError(
            "Anchor pair count mismatch after validation. "
            f"Length of source pairs {len(src_pairs)}, "
            f"length of target pairs {len(tgt_pairs)}"
        )

    model_tokens: List[str] = []
    tags: List[MultiHeadTag] = []

    # Handle prefix structure tokens before first anchor
    prefix_actions, bos_adds = segment_to_actions(src_prefix, tgt_prefix)

    model_tokens.extend(src_prefix)
    tags.extend(prefix_actions)

    # If target has extra prefix additions, attach them to first available token
    if bos_adds:
        if not tags:
            raise ValueError(
                f"Prefix additions {bos_adds} exist but there is no token to attach them to."
            )
        attach_add(tags, 0, bos_adds)

    # Handle anchored segments
    for (src_anchor, src_seg), (tgt_anchor, tgt_seg) in zip(src_pairs, tgt_pairs):
        if src_anchor != tgt_anchor:
            raise ValueError(f"Anchor mismatch: {src_anchor!r} vs {tgt_anchor!r}")

        # Anchor token itself: always KEEP initially
        model_tokens.append(src_anchor)
        tags.append(new_keep_tag())

        seg_actions, seg_prefix_adds = segment_to_actions(src_seg, tgt_seg)

        # Tokens inserted before this segment are attached to the anchor
        if seg_prefix_adds:
            attach_add(tags, len(tags) - 1, seg_prefix_adds)

        model_tokens.extend(src_seg)
        tags.extend(seg_actions)

    assert len(model_tokens) == len(tags)

    return {
        "tokens": model_tokens,
        "op": [t.op for t in tags],
        "replace": [t.replace for t in tags],
        "add": [t.add for t in tags],
    }