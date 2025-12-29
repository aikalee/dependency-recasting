import torch
from stanza.models.common.utils import unsort
from stanza.utils.conll import CoNLL
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.depparse.data import DataLoader
from stanza.models.depparse.trainer import Trainer


def inference_pipeline(model_file, shorthand, input_path, output_path):

    def model_init(model_file):
        print("Initializing model...")
        cache = FoundationCache()
        trainer = Trainer(pretrain=None, model_file=model_file, device="cuda", foundation_cache=cache)
        return trainer

    def preprocess(input_path, shorthand):
        doc = CoNLL.conll2doc(input_path)
        args = {
            'shorthand': shorthand,
            'pretrain': False,
            'sample_train': 1.0,
            }
        batch = DataLoader(doc, batch_size=5000, args=args, pretrain=None, vocab=None, evaluation=True,
                    sort_during_eval=True,
                    min_length_to_batch_separately=200)
        return doc, batch
    
    def parse_document(trainer, doc, batch):
        try:
            with torch.no_grad():
                preds = []
                for i, b in enumerate(batch):
                    preds += trainer.predict(b)
            if batch.data_orig_idx is not None:
                preds = unsort(preds, batch.data_orig_idx)
            batch.doc.set((doc.HEAD, doc.DEPREL), [y for x in preds for y in x])
            # build dependencies based on predictions
            for sentence in batch.doc.sentences:
                sentence.build_dependencies()
            return batch.doc
        except RuntimeError as e:
            if str(e).startswith("CUDA out of memory. Tried to allocate"):
                new_message = str(e) + " ... You may be able to compensate for this by separating long sentences into their own batch with a parameter such as depparse_min_length_to_batch_separately=150 or by limiting the overall batch size with depparse_batch_size=400."
                raise RuntimeError(new_message) from e
            else:
                raise
    
    trainer = model_init(model_file)
    doc, batch = preprocess(input_path, shorthand)
    preds = parse_document(trainer, doc, batch)
    CoNLL.write_doc2conll(preds, output_path)

def main():
    SHORTHAND = 'en_ptb'
    MODEL_FILE = f'/root/autodl-tmp/recasting/depparse-models/UD_English-Penn/{MODELNAME}/{SHORTHAND}_transformer_parser_checkpoint.pt"'
    INPUT_FILE = '/root/autodl-tmp/recasting/data/UD_English-Penn/en_penn-test.conllu'
    OUTPUT_FILE = '/root/autodl-tmp/recasting/predictions/'
    inference_pipeline()

    