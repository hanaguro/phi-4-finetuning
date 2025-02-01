#!/usr/bin/env python

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt, get_chat_template, train_on_responses_only

DATA_FILE="merged_sharegpt.json"
max_seq_length = 2048
#max_seq_length = 1024   # 2048→1024に下げるとさらにメモリ削減
#max_seq_length = 512
load_in_4bit = True     # 4bit量子化でメモリ削減
#load_in_4bit = False
load_in_8bit = False
#fp16 = True             # FP16を有効化（GPUによってはbfloat16でもよい）
fp16 = False
bf16 = True
max_steps = 1000

# ---------- 1) GPUでモデルロード (4bit) ----------
model, tokenizer = FastLanguageModel.from_pretrained(
#    model_name="unsloth/Phi-4",
    model_name="./Phi-4",
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,  # 4bit量子化
    load_in_8bit=load_in_8bit,
    device_map="auto",          # 自動的にGPUを使う
    # fast_inference=True でもOKですが、最初はオフにして様子見でも可
)

# LoRAアダプターを接続
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank (さらに下げる(=8など)と少しメモリ削減になる)
#    r=8,  # LoRA rank
#    r=32,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # gradient checkpointingでさらにメモリ削減
    random_state=3407,
)

# データセットをロード
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = standardize_sharegpt(dataset)

# チャットテンプレートを使用
tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in examples["conversations"]
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# ---------- 2) TrainingArgumentsをGPU用に ----------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=1,  # バッチサイズを小さくしてメモリ節約
        gradient_accumulation_steps=4,  # 大きなバッチを模擬
        warmup_steps=5,
        max_steps=max_steps,
#        max_steps=10,
#        max_steps=1000,
#        learning_rate=2e-4,
        learning_rate=5e-5,
        fp16=fp16,               # True なら半精度を有効化 (GPUメモリ削減)
        bf16=bf16,              # GPUがbfloat16対応ならTrueでもOK
        logging_steps=1,
        optim="adamw_8bit",      # 8bit オプティマイザでメモリ削減
        weight_decay=0.01,
        output_dir="outputs",
        report_to="none",
        # no_cuda は指定しない (GPUを使うため)
        gradient_checkpointing=True,
    ),
)

# ユーザ入力をマスクする (SFT用)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>",
)

FastLanguageModel.for_inference(model)

### トレーニング実行
trainer.train()

###
# ===== 推論 =====
###
# (以下は任意：学習後の動作確認例)

#messages = [
#    {"role": "user", "content": "Plamo Linuxの代表者は誰ですか?"},
#]
#
#inputs = tokenizer.apply_chat_template(
#    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
#)
#
## inputs を GPU に移す
##inputs = {k: v.to(model.device) for k, v in inputs.items()}
#inputs = inputs.to(model.device)
#outputs = model.generate(
##    **inputs,
#    input_ids=inputs,
#    max_new_tokens=1024,
#    use_cache=True,
#    temperature=1.5,
#    min_p=0.1
#)
#
#print(tokenizer.batch_decode(outputs))

# LoRAモデルの保存
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# llama.cpp 用などにGGUF形式で保存（GPU推論後でもOK）
model.save_pretrained_gguf("model", tokenizer)

