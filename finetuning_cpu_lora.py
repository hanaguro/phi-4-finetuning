import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer, IntervalStrategy
from datasets import Dataset, load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# モデルとトークナイザの選択
model_checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# LoRA設定
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,  # ランク
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())

# データセットの準備
df = pd.read_json('merged_sharegpt.json')

# トークナイズ関数の修正
def tokenize_function(examples):
    combined_conversations = []
    for conversation in examples['conversations']:
        # 各辞書から 'content' キーを取り出して結合
        contents = [message['value'] for message in conversation]
        combined_conversation = " ".join(contents)
        combined_conversations.append(combined_conversation)
    
    tokenized_inputs = tokenizer(combined_conversations, truncation=True, padding='max_length', max_length=512)
    # ラベルの仮想的な追加（適切なラベル列を用意してください）
    labels = [0] * len(combined_conversations)  # ここで適切なラベルリストに変更してください
    tokenized_inputs['labels'] = labels
    
    return tokenized_inputs

dataset = Dataset.from_pandas(df)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# データセットの形式を設定
train_dataset = tokenized_datasets.train_test_split(test_size=0.1)['train']
eval_dataset = tokenized_datasets.train_test_split(test_size=0.1)['test']

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# TrainingArgumentsの設定
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy=IntervalStrategy.EPOCH,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainerの初期化前にモデルをCPUに移動
model.to("cpu")

# Trainerの初期化（tokenizerの代わりにprocessing_classを使用）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,  # この行は一時的に残しておきますが、将来は削除または変更が必要です
)

# 訓練の実行
trainer.train()

# トレーニング後のモデルとLoRAパラメータを保存
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# または、peftの関数を使用して保存することも可能です
model = model.merge_and_unload()  # LoRAパラメータをベースモデルにマージ
model.save_pretrained("./fine_tuned_model_merged")
tokenizer.save_pretrained("./fine_tuned_model_merged")
