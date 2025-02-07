import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 必要に応じて変更してください
import pandas as pd
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer, IntervalStrategy
from datasets import Dataset, load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.preprocessing import LabelEncoder

# モデルとトークナイザの選択
model_checkpoint = "./Phi-4"
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=10)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# LoRA設定
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,  # ランク
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())

# データセットの準備
df = pd.read_json('merged_sharegpt.json')
dataset = Dataset.from_pandas(df)

# トレーニングデータセットのカラム名を表示
#print(dataset['conversations'].column_names)

# ラベル列の一覧を取得
labels = dataset['label']
unique_labels = set(labels)
print(f"Unique labels in the dataset: {unique_labels}")

# すべてのラベルを含むリストを作成する
all_labels = list(unique_labels)

# LabelEncoder を初期化し、全てのラベルでフィットさせる
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# LabelEncoderをインスタンス化し、全体のデータセットに対してfit_transformを呼び出す
# label_encoder = LabelEncoder()
# df['label'] = label_encoder.fit_transform(df['label'])

def tokenize_function(examples):
    # 各会話の value を取り出して連結する
    texts = [" ".join(conv['value'] for conv in example) for example in examples['conversations']]
    tokenized_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=512)
    # ラベルをエンコードする
    labels = label_encoder.transform(examples['label'])
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# データセットの形式を設定
split_dataset = tokenized_datasets.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

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

# Trainerの初期化（tokenizerの代わりにprocessing_classを使用）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 訓練の実行
trainer.train()

# トレーニング後のモデルとLoRAパラメータを保存
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# LoRAパラメータをベースモデルにマージ
model = model.merge_and_unload()
model.save_pretrained("./fine_tuned_model_merged")
tokenizer.save_pretrained("./fine_tuned_model_merged")
