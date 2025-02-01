#!/usr/bin/env python

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt, get_chat_template, train_on_responses_only


### FastLanguageModel.from_pretrained()
max_seq_length = 2048   # モデルが処理できる最大トークン数
                        # 2048→1024に下げるとさらにメモリ削減
load_in_4bit = True     # 4bit量子化でメモリ削減
load_in_8bit = False
###

### FastLanguageModel.get_peft_model()
lora_rank = 16          # 低くするとメモリ節約になるが、学習能力が落ちる
lora_alpha = 16         # 大きすぎると過学習しやすく、学習が不安定になる可能性がある
                        # 小さすぎると LoRA の影響が小さくなり、十分な適応が行われない
lora_dropout = 0.1        # LoRA の学習時にニューロンをランダムに無効化し、過学習を防ぐための手法
bias = "none"           # "none" メモリ節約と安定した学習が可能。
                        # "all" メモリ消費は増えるが、柔軟な適応が可能
                        # "lora_only" バランス型。
use_gradient_checkpointing="unsloth"    # メモリ削減。計算速度は落ちる
random_state = 3407     # 再現性のためのランダムシード
###

### load_dataset()
DATA_FILE="merged_sharegpt.json"
###

### SFTTrainer()
max_steps = 1000
fp16 = True            # FP16を有効化（GPUによってはbfloat16でもよい）
bf16 = False
###

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
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias=bias,
    use_gradient_checkpointing=use_gradient_checkpointing,   
                                            
    random_state=random_state,
)

# データセットをロード
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = standardize_sharegpt(dataset)

# チャットテンプレートを使用
# 指定したモデル (Phi-4) に適した会話フォーマットを設定する
tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

# 各サンプルに対して会話フォーマットを適用し、モデルが処理しやすいデータに変換する
def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(convo, 
            tokenize=False,   # トークナイズをせずに、そのままテキストとして返す
            add_generation_prompt=False)  # 追加のプロンプト (<|endoftext|> など) を付けず、純粋なフォーマット変換のみを行う
        for convo in examples["conversations"]
    ]
    return {"text": texts}

# データセットの全てのサンプルに formatting_prompts_func() を適用
dataset = dataset.map(formatting_prompts_func, 
                        batched=True    # 複数のデータをまとめて処理し、速度を最適化
                      )

# ---------- 2) TrainingArgumentsをGPU用に ----------
# モデルに対する教師ありファインチューニングの設定と準備
trainer = SFTTrainer(
    model=model,    # FastLanguageModel.get_peft_model() で作成した LoRA 適用済み
    tokenizer=tokenizer,    # get_chat_template() でフォーマット調整済み
    train_dataset=dataset,  # formatting_prompts_func() を適用済み
    dataset_text_field="text",  # "text" フィールド (会話データ) をモデルの入力として学習
    max_seq_length=max_seq_length,  # モデルの最大シーケンス長
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),  # DataCollatorForSeq2Seq は、トークナイズ後のデータを 適切なバッチにまとめる処理 を担当
    dataset_num_proc=2, # データセットの並列処理数
    args=TrainingArguments(  # トレーニングの詳細な設定を指定
        per_device_train_batch_size=1,  # バッチサイズを小さくしてメモリ節約
        gradient_accumulation_steps=4,  # 大きなバッチを模擬(4でバッチサイズ4に相当するトレーニングを実行可能)
        warmup_steps=5, # 最初の 5 ステップは小さい学習率でトレーニング。過学習を防ぐための設定
        max_steps=max_steps, # トレーニングを 最大 max_steps ステップまで実行
#        learning_rate=2e-4,
        learning_rate=5e-5, # 学習率
        fp16=fp16,              # True なら半精度を有効化 (GPUメモリ削減)
        bf16=bf16,              # GPUがbfloat16対応ならTrueでもOK
        logging_steps=1,        # 1 ステップごとにログを出力
        optim="adamw_8bit",     # 8bit オプティマイザでメモリ削減
        weight_decay=0.01,      # 過学習を防ぐための重み減衰 (L2 正則化)
        output_dir="outputs",   # 学習済みモデルやログをディレクトリに保存
        report_to="none",       # WandB (Weights & Biases) などのログサービスを使わない設定
        # no_cuda は指定しない (GPUを使うため)
        gradient_checkpointing=True,    # 勾配チェックポイントを有効化。VRAM 使用量を削減できるが、計算量は増加
    ),
)

# ユーザ入力をマスクする (SFT用)
# この関数は、ユーザーの入力 (instruction_part) を無視し、アシスタントの応答 (response_part) のみを学習させるための処理を行う
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>",
)

# ファインチューニングを開始
trainer.train()

# モデルを推論専用のモードに切り替えるための設定を行う
FastLanguageModel.for_inference(model)

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
