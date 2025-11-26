import tarfile

def read_conllu_file_from_tgz(tgz_path, target_filename):
    """
    Parses one .conllu file from a .tgz archive and yields each sentence.
    """
    
    with tarfile.open(tgz_path, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(target_filename)), None)
        if member is None:
            raise FileNotFoundError(f"{target_filename} not found in archive.")

        f = tar.extractfile(member)
        if f is None:
            raise IOError(f"Could not extract {target_filename} from archive.")

        lines = f.read().decode("utf-8").splitlines()
        sentence = {}
        for line in lines:
            line = line.strip()
            if not line:
                if sentence:
                    yield sentence
                    sentence = {}
            elif not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) == 10:
                    if '-' in parts[0] or '.' in parts[0]:
                        continue
                    token = {
                        'id': int(parts[0]),
                        'form': parts[1],
                        # 'lemma': parts[2],
                        "UPOS": parts[3],
                        # 'XPOS': parts[4],
                        # 'feats': parts[5] if parts[5] != '_' else None,
                        'head': int(parts[6]),
                        'deprel': parts[7],
                    }
                    sentence[token['id']] = token
        if sentence:
            yield sentence

def read_conllu_file(file_path, return_sent_ids=False, return_text=False):
    """
    Parses a .conllu file and yields each sentence.
    Input:  A .conllu file path.
    Output: Yields sentences as dictionaries, optionally with sentence IDs and text.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        sentence = {}
        sent_id = None
        text = None
        for line in lines:
            line = line.strip()
            if not line:
                if sentence:
                    if return_sent_ids and return_text:
                        yield (sentence, sent_id, text)
                    elif return_sent_ids:
                        yield (sentence, sent_id)
                    elif return_text:
                        yield (sentence, text)
                    else:
                        yield sentence
                    sentence = {}
                    sent_id = None
                    text = None
            elif line.startswith("# sent_id"):
                sent_id = line.split('=')[1].strip()
            elif line.startswith("# text"):
                text = line.split('=')[1].strip()
            elif not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) == 10:
                    if '-' in parts[0] or '.' in parts[0]:
                        continue
                    token = {
                        'id': int(parts[0]),
                        'form': parts[1],
                        'lemma': parts[2],
                        'UPOS': parts[3],
                        'XPOS': parts[4],
                        'feats': parts[5],
                        'head': int(parts[6]),
                        'deprel': parts[7],
                        'deps': parts[8],
                        'misc': parts[9],
                    }
                    sentence[token['id']] = token
        
        if sentence:
            if return_sent_ids and return_text:
                yield (sentence, sent_id, text)
            elif return_sent_ids:
                yield (sentence, sent_id)
            elif return_text:
                yield (sentence, text)
            else:
                yield sentence


def sentence_to_conllu(sentence, sent_id="0", text=None):
    """
    Converts a sentence dictionary to a CoNLL-U formatted string.
    Input:  A sentence dictionary where keys are token IDs and values are dictionaries
                containing token attributes like 'form', 'lemma', 'UPOS', etc.
    Output: A string representing the sentence in CoNLL-U format.
    """


    conllu_lines = [f"# sent_id = {sent_id}"]
    
    if text is not None:
        conllu_lines.append("# text = " + text)
    else:
        conllu_lines.append("# text = " + ' '.join(tok.get('form', '_') for tok in sorted(sentence.values(), key=lambda x: x.get('id', 0))))


    # print(f"Sentence ID: {sent_id}")
    for i, tok in sorted(sentence.items(), key=lambda x: x[0]):
        if i == 0:
            continue
        conllu_line = [
            str(tok.get('id', i)),                 # ID
            tok.get('form', '_'),                  # FORM
            tok.get('lemma', '_'),                 # LEMMA
            tok.get('UPOS', '_'),                  # UPOS
            tok.get('XPOS', '_'),                  # XPOS
            tok.get('feats', '_'),                 # FEATS
            str(tok.get('head', 0)),               # HEAD
            tok.get('deprel', '_'),                # DEPREL
            tok.get('deps', '_'),                  # DEPS
            tok.get('misc', '_')                   # MISC
        ]
        # print(f"\t{i}\t{conllu_line}")
        
        conllu_lines.append('\t'.join(conllu_line))
    
    return '\n'.join(conllu_lines)+ '\n'

# def sentences_to_conllu(sentences):
#     """
#     Converts a list of sentences to a CoNLL-U formatted string.
#     Input:  A list of sentence dictionaries.
#     Output: A string representing the sentences in CoNLL-U format.
#     """
#     return '\n'.join(sentence_to_conllu(sentence, sent_id) for sent_id, sentence in enumerate(sentences)) + '\n'

def sentences_to_conllu(sentences, sent_ids=None, text_lst=None):
    """
    Converts a list of sentences to a CoNLL-U formatted string.
    Input:  A list of sentence dictionaries and optional list of sentence IDs and optional list of texts.
    Output: A string representing the sentences in CoNLL-U format.
    """
    if sent_ids is not None and len(sent_ids) != len(sentences):
        raise ValueError("Length of sent_ids must match length of sentences.")
    if text_lst is not None and len(text_lst) != len(sentences):
        raise ValueError("Length of text_lst must match length of sentences.")
    
    conllu_lines = []
    for i, sentence in enumerate(sentences):
        sent_id = sent_ids[i] if sent_ids else str(i + 1)
        text = text_lst[i] if text_lst else None
        conllu_lines.append(sentence_to_conllu(sentence, sent_id, text))
    
    return '\n'.join(conllu_lines) + '\n'

def write_conllu_file(file_path, sentences, sent_ids=None, text_lst=None):
    """
    Writes a list of sentences to a CoNLL-U formatted file.
    Input:  A list of sentence dictionaries, file path, and optional lists of sentence IDs and texts.
    Output: Writes the sentences to the specified file in CoNLL-U format.
    """
    conllu_string = sentences_to_conllu(sentences, sent_ids, text_lst)
    with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(conllu_string)

def read_eval_table(file_path):
    """
    Reads an evaluation table generated by conll18_ud_eval.py and returns a dictionary of metrics.
    Input: A file path to the evaluation table.
    Output: A nested dictionary where each outer key is a metric type (e.g., "Precision"),
              and each inner key is a metric name (e.g., "Tokens"), mapping to a float value.
              For example: dictionary["Precision"]["Tokens"] = 100.0
    
    """
    
    
    with open(file_path, 'r', encoding='utf-8') as f:
        table_str = f.read()
    
    lines = table_str.strip().split('\n')
    headers = [h.strip() for h in lines[0].split('|')[1:]]  # Skip "Metric"
    data = {}

    for line in lines[2:]:  # Skip header and separator lines
        parts = [p.strip() for p in line.split('|')]
        metric = parts[0]
        values = parts[1:]

        for i, value in enumerate(values):
            if headers[i] not in data:
                data[headers[i]] = {}
            if value:  # Only store non-empty values
                data[headers[i]][metric] = float(value)

    return data

def main():
    # folder_path = "../data/ud-treebanks-v2.15.tgz"
    # file_path = "yue_hk-ud-test.conllu"
    # sentences = []
    # count = 0
    # for sentence in read_conllu_file_from_tgz(folder_path, file_path):
    #     count +=1
    #     print("Sentence:", sentence)
    #     print()  # Print a blank line between sentences
    #     sentences.append(sentence)

    #     if count >= 2:
    #         break

    # conllu_string = sentences_to_conllu(sentences)
    # print("CoNLL-U formatted string:")
    # print(conllu_string)


    # file_path = "../../data/UD_Danish-DDT/da_ddt-ud-test.conllu"
    # for sentence in read_conllu_file(file_path, return_sent_ids=True, return_text=True):
    #     print(sentence)
    #     break


    file_path = "../../analysis/validation_da_ddt-ud-test_projectivized_sentences_left_smallest_head_parsed.txt"
    result = read_eval_table(file_path)
    print(result)


if __name__ == "__main__":
    main()