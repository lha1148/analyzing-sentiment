from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Load mô hình đã fine-tuned
model = BertForSequenceClassification.from_pretrained("path/to/checkpoint-255")

# Nếu cần load tiếp để huấn luyện:
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="path/to/output"),
    train_dataset='train_data.csv',
    eval_dataset='val_data'
)

trainer.train(resume_from_checkpoint="results/checkpoint-255")
