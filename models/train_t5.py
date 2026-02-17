import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "predictions" / "downstream"
TRAINED_MODEL_DIR = BASE_DIR / "trained-t5"
RAW_MODEL_PATH = "/root/autodl-tmp/t5-small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Now running at:", DEVICE)

# === Load model ===
def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    return tokenizer, model
     
# === Create dataset ===
def load_data(tokenizer, file, max_length=512):
    with open(file, "r") as f:
        data_list = [line.strip() for line in f.readlines()]
    return tokenizer(data_list, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

class CustomDataset(Dataset):
    def __init__(self, tokenizer, target_path, source_path):
        self.tgt_data = load_data(tokenizer, target_path, 512)
        self.src_data = load_data(tokenizer, source_path, 128)
        self.labels = self.tgt_data["input_ids"]
    
    def __len__(self):
        return self.src_data["input_ids"].size(0)
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.src_data.items()}
        item["labels"] = self.labels[idx]
        return item

def create_dataset(tokenizer, data_dir, split):
    tgt_file = data_dir + f"{split}.tgt.txt"
    src_file = data_dir + f"{split}.src.txt"
    return CustomDataset(tokenizer, tgt_file, src_file)
                        
# === Training ===
def train_model(tokenizer, model, train_dataset, eval_dataset, model_dir):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    args = Seq2SeqTrainingArguments(
        output_dir=model_dir, 
        per_device_train_batch_size=8, 
        predict_with_generate=False,
        save_total_limit=1
    )
    trainer = Seq2SeqTrainer(
        model=model, 
        args=args, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        tokenizer=tokenizer, 
        data_collator=data_collator
    )
    trainer.train()

def predict_dataset(tokenizer, model, test_dataset, output_file):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)

    model.eval()
    with open(output_file, "w", encoding="utf-8") as f, torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            gen_ids = model.generate(**batch, max_new_tokens=128, num_beams=4)
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for line in preds:
                f.write(line + "\n")
    print(f"Done! Predictions written to {output_file}.")

def main():
    mode = "predict"
    lang = "grc"
    pos = "upos"
    data = f"lang={lang},pos={pos}"
    data_dir = f"/root/autodl-tmp/recasting/data/downstream/{data}/"  

    if mode == "train":
        model_dir = TRAINED_MODEL_DIR / data
        tokenizer, model = load_model(RAW_MODEL_PATH, DEVICE)
        train_dataset = create_dataset(tokenizer, data_dir, "train")
        eval_dataset = create_dataset(tokenizer, data_dir, "dev")
        train_model(tokenizer, model, train_dataset, eval_dataset, model_dir)
        
    if mode == "predict":
        output_file = OUTPUT_DIR / f"{data}.txt"
        lang_model_dir = TRAINED_MODEL_DIR / data
        trained_model = list(lang_model_dir.glob("checkpoint-*"))[0]
        tokenizer, model = load_model(trained_model, DEVICE)
        test_dataset = create_dataset(tokenizer, data_dir, "test")
        predict_dataset(tokenizer, model, test_dataset, output_file)

    
if __name__ == "__main__":
    main()
    





    


    



    
  