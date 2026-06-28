import json
import torch
from tqdm import tqdm
from collections import Counter
from src.data.downstream.preprocessing.combine_datasets import LANGUAGES
from src.pathgen import DIR_ABBR_LOOKUP, get_edit_actions_json_path, get_structured_tokens_json_path, get_vocab_dir

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as fin:
       examples = json.load(fin)
    return examples

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
LABEL_OFFSET = 3


# def build_token_vocab(examples, min_freq=1):
#     counter = Counter()
#     for ex in examples:
#         counter.update(ex["tokens"])

#     token2id = {"<PAD>": 0, "<UNK>": 1}

#     for tok, freq in counter.items():
#         if freq >= min_freq and tok not in token2id:
#             token2id[tok] = len(token2id)

#     id2token = {i: t for t, i in token2id.items()}
#     return token2id, id2token


def build_token_vocab(examples, min_freq=1):
    counter = Counter()
   
    for ex in tqdm(examples, desc="Building token vocab"):          
        for item in ex["target"]["local"]:

            counter[item["token"]] += 1
        # counter[ex["target"]["token"]] += 1

    token2id = {"<PAD>": 0, "<UNK>": 1}

    for tok, freq in counter.items():
        if freq >= min_freq and tok not in token2id:
            token2id[tok] = len(token2id)

    id2token = {i: t for t, i in token2id.items()}
    return token2id, id2token


# def build_op_vocab(examples):
#     counter = Counter()
#     for ex in examples:
#         counter.update(ex["op"])
#     ops = sorted({op for ex in examples for op in ex["op"]})
#     op2id = {op: i for i, op in enumerate(ops)}
#     id2op = {i: op for op, i in op2id.items()}
#     print(f"Count of operations: {counter}")
#     return op2id, id2op


# def build_deprel_vocab(examples):
#     rels = set()

#     for ex in examples:
#         for r in ex["replace"]:
#             if r is not None:
#                 rels.add(r)

#         for add_list in ex["add"]:
#             for r in add_list:
#                 rels.add(r)

#     rels = sorted(rels)
#     deprel2id = {rel: i for i, rel in enumerate(rels)}
#     id2deprel = {i: rel for rel, i in deprel2id.items()}
#     return deprel2id, id2deprel

def get_all_exapmles(langs):
    all_examples = []

    for lang in tqdm(langs, desc="Getting examples from all languages"):
        for split in ["train", "dev"]:
            file_path = get_structured_tokens_json_path(lang, pos="upos", split=split)
            all_examples.extend(read_jsonl(file_path))

    return all_examples

def build_lang_vocab(langs):
    lang2id = {"<unk>": 0}
    
    for i, lang in enumerate(langs, start=1):
        abbr = DIR_ABBR_LOOKUP[lang]
        lang2id[abbr] = i
    id2lang = {i: lang for lang, i in lang2id.items()}
    return lang2id, id2lang
        


def build_bracket_vocab(examples):
    rels = set()

    for ex in tqdm(examples, desc="Building bracket vocab"): 

        for src_item, tgt_item in zip(ex["source"], ex["target"]["local"]):
            
                # if isinstance(src_item, list):
                #     print("is list", src_item)
                rels.update(src_item["left"])
                rels.update(src_item["right"])
                rels.update(tgt_item["left"])
                rels.update(tgt_item["right"])

    rels = sorted(rels)
    bracket2id = {PAD: PAD_ID, BOS: BOS_ID, EOS: EOS_ID}

    for i, rel in enumerate(rels, start=LABEL_OFFSET):
        bracket2id[rel] = i
    # left2id = {rel: i for i, rel in enumerate(rels)}
    id2bracket = {i: rel for rel, i in bracket2id.items()}
    return bracket2id, id2bracket



def build_left_vocab(examples):
    rels = set()
    

    for ex in tqdm(examples, desc="Building left vocab"):          # ex = one sentence (list of token dicts)
        # for src, tgt in zip(ex["source"], ex["target"]):
        
        for src_item, tgt_item in zip(ex["source"], ex["target"]["local"]):
           
            # if isinstance(src_item, list):
            #     print("is list", src_item)
            rels.update(src_item["left"])
            rels.update(tgt_item["left"])

    rels = sorted(rels)
    left2id = {PAD: PAD_ID, BOS: BOS_ID, EOS: EOS_ID}

    for i, rel in enumerate(rels, start=LABEL_OFFSET):
        left2id[rel] = i
    # left2id = {rel: i for i, rel in enumerate(rels)}
    id2left = {i: rel for rel, i in left2id.items()}
    return left2id, id2left


def build_right_vocab(examples):
    rels = set()

    for ex in tqdm(examples, desc="Building right vocab"):          # ex = one sentence (list of token dicts)
        # for item in ex:
        for src_item, tgt_item in zip(ex["source"], ex["target"]["local"]):
            rels.update(src_item["right"])
            rels.update(tgt_item["right"])

    rels = sorted(rels)
    right2id = {PAD: PAD_ID, BOS: BOS_ID, EOS: EOS_ID}

    for i, rel in enumerate(rels, start=LABEL_OFFSET):
        right2id[rel] = i

    # right2id = {rel: i for i, rel in enumerate(rels)}
    id2right = {i: rel for rel, i in right2id.items()}
    return right2id, id2right

def build_decoder_vocab(examples):
    rels = set()

    for ex in tqdm(examples, desc="Building decoder vocab"):          # ex = one sentence (list of token dicts)
        # for item in ex:
        rels.update(ex["target"]["global"])

    rels = sorted(rels)
    decoder2id = {PAD: PAD_ID, BOS: BOS_ID, EOS: EOS_ID}

    for i, rel in enumerate(rels, start=LABEL_OFFSET):
        decoder2id[rel] = i

    # right2id = {rel: i for i, rel in enumerate(rels)}
    id2decoder = {i: rel for rel, i in decoder2id.items()}
    return decoder2id, id2decoder


def build_vocabs_from_datasets(langs, save_dir=get_vocab_dir()):
    all_examples = get_all_exapmles(langs)

    lang2id, id2lang = build_lang_vocab(LANGUAGES)
    token2id, id2token = build_token_vocab(all_examples)
    # op2id, id2op = build_op_vocab(all_examples)
    # deprel2id, id2deprel = build_deprel_vocab(all_examples)
    bracket2id, id2bracket = build_bracket_vocab(all_examples)
    left2id, id2left = build_left_vocab(all_examples)
    right2id, id2right = build_right_vocab(all_examples)

    
    # decoder2id, id2decoder = build_decoder_vocab(all_examples)

    vocab_dict = {
        "lang2id": lang2id,
        "id2lang": id2lang,
        "token2id": token2id,
        "id2token": id2token,
        "bracket2id": bracket2id,
        "id2bracket": id2bracket,
        "left2id": left2id,
        "id2left": id2left,
        "right2id": right2id,
        "id2right": id2right,
        # "decoder2id": decoder2id,
        # "id2decoder": id2decoder,
        # "op2id": op2id,
        # "id2op": id2op,
        # "deprel2id": deprel2id,
        # "id2deprel": id2deprel,
    }
    return vocab_dict

# def build_replace_class_weight(examples, save_dir=get_vocab_dir()):
#     counter = Counter()


#     deprel2id, id2deprel = build_deprel_vocab(examples)

#     for ex in examples:
#         for r in ex["replace"]:
#             if r is not None:
#                 counter[r] += 1

#     total = sum(counter.values())
#     num_classes = len(deprel2id)

#     weights = torch.ones(num_classes, dtype=torch.float)

#     for rel, idx in deprel2id.items():
#         freq = counter.get(rel, 1)
#         weights[idx] = total / (num_classes * freq)
    
#     weights = weights / weights.mean()
#     weights = torch.clamp(weights, max=10.0)
#     weight_list = weights.tolist()
#     return {"replace_class_weight": weight_list}



def build_pos_weight(examples, label2id, side: str, max_weight: float = 10.0):
    """
    examples: list[list[dict]]
    side: "left" or "right"

    Computes BCEWithLogitsLoss pos_weight:
        pos_weight = num_negative / num_positive

    Then normalizes and clamps for stability.
    """
    num_labels = len(label2id)

    pos_counts = torch.zeros(num_labels, dtype=torch.float)
    total_positions = 0


    for ex in tqdm(examples, desc=f"Counting {side} labels"):
        for item in ex["target"]["local"]:
            total_positions += 1
            for label in item[side]:
                pos_counts[label2id[label]] += 1

    neg_counts = total_positions - pos_counts

    # avoid division by zero
    pos_counts = torch.clamp(pos_counts, min=1.0)

    pos_weight = neg_counts / pos_counts

    # stabilize scale
    pos_weight = pos_weight / pos_weight.mean()
    pos_weight = torch.clamp(pos_weight, max=max_weight)

    return {f"{side}_pos_weight": pos_weight.tolist()}

def build_class_weight(examples, side: str, max_weight: float = 10.0):
    counter = Counter()

    for ex in tqdm(examples, desc=f"Counting {side} decoder label"):
        for token in ex["target"]["local"]:
            counter.update(token[side])
    
    total_freq = sum(counter.values())
    num_classes = len(counter)
    weights = torch.ones(num_classes+3, dtype=torch.float)

    for idx, (rel, freq) in enumerate(counter.items(), start=3):
        weights[idx] = total_freq / (num_classes * freq)
    
    weights = weights / weights.mean()
    weights = torch.clamp(weights, max=max_weight)

    weights[0] = 0.0
    weights[1] = 0.0
    weights[2] = 0.2

    weight_list = weights.tolist()
    return {f"{side}_class_weight": weight_list}

def build_global_class_weight(examples,  max_weight: float = 10.0):
  
    counter = Counter()

    for ex in tqdm(examples, desc="Counting decoder input"):
        counter.update(ex["target"]["global"])

    total_freq = sum(counter.values())
    num_classes = len(counter)
    weights = torch.ones(num_classes, dtype=torch.float)

    for idx, (rel, freq) in enumerate(counter.items()):
        weights[idx] = total_freq / (num_classes * freq)
    
    weights = weights / weights.mean()
    weights = torch.clamp(weights, max=max_weight)
    weight_list = weights.tolist()
    return {"global_class_weight": weight_list}


def write_to_json(langs, max_weight=10, save_dir=get_vocab_dir()):
    all_examples = get_all_exapmles(langs)
    vocab_dict = build_vocabs_from_datasets(langs)
    # replace_weights = build_replace_class_weight(langs)
    # vocab_and_weights = vocab_dict | replace_weights

    # left2id = vocab_dict["left2id"]
    # right2id = vocab_dict["right2id"]

    left_decoder_class_weight = build_class_weight(
        all_examples,
        side="left",
        max_weight=max_weight,
    )

    right_decoder_class_weight = build_class_weight(
        all_examples,
        side="right",
        max_weight=max_weight,
    )



    vocab_and_weights = vocab_dict | left_decoder_class_weight | right_decoder_class_weight

    with open(save_dir / f"vocab_and_weight.json", "w", encoding="utf-8") as fout:
        json.dump(vocab_and_weights, fout, indent=4)
    print(f"Done! Vocab written to {save_dir}")

