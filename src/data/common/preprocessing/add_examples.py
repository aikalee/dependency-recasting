from src.data.common.conllu_io import read_conllu
from tqdm import tqdm

def add_examples(train_file, dev_file, train_recording_file, dev_recording_file):
    all_train_labels = []
    all_dev_labels = []
    train_ex_to_add = []
    dev_ex_to_add = []
    # with open(dev_file, "r", encoding="utf-8") as fdev:
    for tokenlist, sentencdata in tqdm(read_conllu(train_file), desc="Collecting train labels"):
        for token in tokenlist:
            all_train_labels.append(token["deprel"])

    for tokenlist, sentencdata in tqdm(read_conllu(dev_file), desc="Collecting dev labels"):
        for token in tokenlist:
            all_dev_labels.append(token["deprel"])
                
    # with open(train_file, "r", encoding="utf-8") as ftrain:
    for tokenlist, sentencedata in tqdm(read_conllu(dev_file), desc="Adding to dev"):
        flag = False
        for token in tokenlist:
            if not token["deprel"] in all_train_labels:
                flag = True
        if flag:
            dev_ex_to_add.append(tokenlist.serialize())

    for tokenlist, sentencedata in tqdm(read_conllu(train_file), desc="Adding to dev"):
        flag = False
        for token in tokenlist:
            if not token["deprel"] in all_dev_labels:
                flag = True
        if flag:
            train_ex_to_add.append(tokenlist.serialize())
    
    with open(train_file, "a", encoding="utf8") as ftrain, \
         open(dev_file, "a", encoding="utf-8") as fdev, \
         open(train_recording_file, "w", encoding="utf-8") as ftrain_recording, \
         open(dev_recording_file, "w", encoding="utf-8") as fdev_recording:
        ftrain_recording.write("\n".join(dev_ex_to_add))
        fdev_recording.write("\n".join(train_ex_to_add))
        ftrain.write("\n".join(dev_ex_to_add))
        fdev.write("\n".join(train_ex_to_add))



