import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

from conllu import parse_incr
from stanza.models.pos.data import Dataset
from stanza.models.common.doc import Document
from stanza.models.common.pretrain import Pretrain
from stanza.models.pos.trainer import Trainer as POSTrainer
from stanza.models.constituency.trainer import Trainer as ConstTrainer

from pathlib import Path
from tqdm import tqdm

def write_to_file(predicted_trees, output_path):

        # === Create/clear output file ===
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fout:
            for pred in predicted_trees:
                tree = pred.predictions[0].tree
                fout.write(str(tree))
                fout.write("\n")
    
        print(f"Done! Predictions written to {output_path}")

def retag_pipeline(input_path, output_path, pos_model_file, pretrain_file, forward_charlm_file, backward_charlm_file, const_model_file):

    def build_sentences(input_path):

        sentences = []

        with open(input_path, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                sentence = []
                token_id = 1
                tokens = line.split()
                for token in tokens:
                    sentence.append({"id": (token_id,), "text": token})
                    token_id += 1
                sentences.append(sentence)

        return sentences
    
    def model_init(pos_model_file, pretrain, forward_charlm_file, backward_charlm_file, const_model_file):
        
        pos_trainer = POSTrainer(
            model_file=pos_model_file,     # this triggers automatic loading
            pretrain=pretrain,
            args={
                "device": "cuda",
                "charlm_forward_file": forward_charlm_file,
                "charlm_backward_file": backward_charlm_file,
            },
            device="cuda"
            )
    
        const_trainer = ConstTrainer.load(const_model_file, args={"device": "cuda"})
        const_model = const_trainer.model
        const_model.eval()

        return pos_trainer, const_model
    
    def preprocess(input_path, pretrain, pos_trainer):

        sentences = build_sentences(input_path)
        doc = Document(sentences)
        dataset = Dataset(
            doc, 
            args={
                'shorthand': 'custom',
                'pretrain': True,
                'bert_model': None,
                'word_cutoff': 0,
                'augment_nopunct': 0.0,
            },
            pretrain=pretrain,
            vocab=pos_trainer.vocab,      
            evaluation=True               # disables masking
        )
        loader = dataset.to_loader(batch_size=32, shuffle=False)
        return doc, loader
    
    def parse_sentences(doc, loader, pos_trainer, const_model):

        upos_tags = []

        for batch in tqdm(loader, desc="POS tagging"):
            preds = pos_trainer.predict(batch, unsort=True) 
            upos = [[t[0] for t in sent] for sent in preds]
            upos_tags.extend(upos)

        tokens = [[w.text for w in sent.words] for sent in doc.sentences]
        tagged_sentences = [list(zip(words, tags)) for words, tags in zip(tokens, upos_tags)]

        predicted_trees = const_model.parse_sentences_no_grad(
            data_iterator=iter(tqdm(tagged_sentences, desc="Constituency parsing")), 
            build_batch_fn=const_model.build_batch_from_tagged_words, 
            batch_size=8, 
            transition_choice=const_model.predict, 
            keep_scores=False
            )
        return predicted_trees

    pretrain = Pretrain(pretrain_file)
    pos_trainer, const_model = model_init(pos_model_file, pretrain, forward_charlm_file, backward_charlm_file, const_model_file)
    doc, loader = preprocess(input_path, pretrain, pos_trainer)
    predicted_trees = parse_sentences(doc, loader, pos_trainer, const_model)
    write_to_file(predicted_trees, output_path)

def no_retag_pipeline(input_path, output_path, const_model_file, pos):
    """
    Input file: CoNLL-U file
    Output file: Predicted PTB file
    """
    def get_gold_pos(input_path, pos):
        tagged_sentences = []
        with open(input_path, "r", encoding="utf-8") as fin:
            for tokenlist in parse_incr(fin):
                sentence = []
                for token in tokenlist:
                    if isinstance(token["id"], int):                  # added Feb 24
                        sentence.append((token["form"].replace(" ", ""), token[pos]))
                tagged_sentences.append(sentence)
        return tagged_sentences
    
    def model_init(const_model_file):
        print("Initializing model...")
        const_trainer = ConstTrainer.load(const_model_file, args={"device": "cuda"})
        const_model = const_trainer.model
        const_model.eval()
        return const_model
    
    def parse_sentences(const_model, tagged_sentences):
        predicted_trees = const_model.parse_sentences_no_grad(
            data_iterator=iter(tqdm(tagged_sentences, desc="Constituency parsing")), 
            build_batch_fn=const_model.build_batch_from_tagged_words, 
            batch_size=8, 
            transition_choice=const_model.predict, 
            keep_scores=False
            )
        return predicted_trees
    
    tagged_sentences = get_gold_pos(input_path, pos.lower())
    const_model = model_init(const_model_file)
    predicted_trees = parse_sentences(const_model, tagged_sentences)
    write_to_file(predicted_trees, output_path)

def main():
    UD_ABBR_LOOKUP = {
        "Ancient_Greek": "grc",
        "Chinese": "zh",
        "English-Penn": "en",
        "English-EWT": "en",
        "Finnish": "fi",
        "French": "fr",
        "Hebrew": "he",
        "Russian": "ru",
        "Tamil": "ta",
        "Uyghur": "ug",
        "Wolof": "wo"
        }
    
    STNZ_ABBR_LOOKUP = {
        "Ancient_Greek": "grc",
        "Chinese": "zh-hans",
        "English-Penn": "en",
        "English-EWT": "en",
        "Finnish": "fi",
        "French": "fr",
        "Hebrew": "he",
        "Russian": "ru",
        "Tamil": "ta",
        "Uyghur": "ug",
        "Wolof": "wo"
        }
    
    TREEBANK_LOOKUP = {
        "Ancient_Greek": "Perseus",
        "Chinese": "Penn",
        "English-Penn": "Penn",
        "English-EWT": "EWT",
        "Finnish": "TDT",
        "French": "GSD",
        "Hebrew": "HTB",
        "Russian": "GSD",
        "Tamil": "TTB",
        "Uyghur": "UDT",
        "Wolof": "WTB"
        }

    
    lang = "Ancient_Greek"
    pos = "upos"
    epochs = 100
    head = "no"
    path = "yes"
    label = f",head={head},path={path}"
    retag = False
    # charlm = False
    ud_abbr = UD_ABBR_LOOKUP[lang]
    stnz_abbr = STNZ_ABBR_LOOKUP[lang]
    treebank = TREEBANK_LOOKUP[lang]
    input_file = f"/root/autodl-tmp/recasting/data/upstream_inference/{ud_abbr}_{treebank.lower()}-ud-test.conllu"


    pretrain_dir = f"/root/autodl-tmp/stanza_resources/{stnz_abbr}/pretrain/"
    pretrain_dir = list(Path(pretrain_dir).glob("*.pt"))[0]

    if retag:
        pos_model_dir = f"/root/autodl-tmp/stanza_resources/{stnz_abbr}/pos/"
        pos_model = list(Path(pos_model_dir).glob("*.pt"))[0]

    # if charlm:
        forward_charlm_dir = f"/root/autodl-tmp/stanza_resources/{stnz_abbr}/forward_charlm/"
        backward_charlm_dir = f"/root/autodl-tmp/stanza_resources/{stnz_abbr}/backward_charlm/"
        forward_charlm = list(Path(forward_charlm_dir).glob("*.pt"))[0]
        backward_charlm = list(Path(backward_charlm_dir).glob("*.pt"))[0]
    
    # pos_model = f"/root/autodl-tmp/stanza_resources/{stnz_abbr}/pos/combined_charlm.pt"
    # pretrain_file = f"/root/autodl-tmp/stanza_resources/{stnz_abbr}/pretrain/conll17.pt"
    # forward_charlm_file = f"/root/autodl-tmp/stanza_resources/{stnz_abbr}/forward_charlm/1billion.pt"
    # backward_charlm_file = f"/root/autodl-tmp/stanza_resources/{stnz_abbr}/backward_charlm/1billion.pt"
   
    model_name = f"lang={stnz_abbr},pos={pos}{label},epochs={epochs}"
    const_model = f"/root/autodl-tmp/recasting/stanza-models/{model_name}/{stnz_abbr}_transformer_finetuned_constituency_checkpoint.pt"
    output_file = f"/root/autodl-tmp/recasting/predictions/stanza/{model_name}.mrg"
    no_retag_pipeline(input_file, output_file, const_model, pos)

if __name__ == "__main__":
    main()

        
    