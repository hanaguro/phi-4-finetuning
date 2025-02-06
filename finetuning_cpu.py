import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer, IntervalStrategy
from datasets import load_dataset

# モデルとトークナイザの選択
model_checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# データセットの準備
dataset = load_dataset("glue", "mrpc")
train_dataset = dataset["train"].map(lambda x: tokenizer(x["sentence1"], x["sentence2"], truncation=True), batched=True)
eval_dataset = dataset["validation"].map(lambda x: tokenizer(x["sentence1"], x["sentence2"], truncation=True), batched=True)

# データセットの形式を設定
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# TrainingArgumentsでデバイスをCPUに指定
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy=IntervalStrategy.EPOCH,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainerの初期化前にモデルをCPUに移動
model.to("cpu")

# Trainerの初期化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 訓練の実行
trainer.train()

# トレーニング後のモデルを保存
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
