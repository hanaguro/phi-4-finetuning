import json

# Alpaca形式のJSONファイルを読み込む
with open("merged.json", "r", encoding="utf-8") as f:
    alpaca_data = json.load(f)

converted_data = []
for i, item in enumerate(alpaca_data):
    # .strip() を適用する前に、明示的に str() に変換
    instruction = str(item.get("instruction", "")).strip()
    input_text = str(item.get("input", "")).strip()
    output_text = str(item.get("output", "")).strip()

    # ユーザのメッセージを構築
    user_message = instruction
    if input_text:
        user_message += f"\n{input_text}"

    # ShareGPT形式の `conversations` に変換
    conversations = [
        {"from": "system", "value": "You are an assistant"},
        {"from": "human", "value": user_message},
        {"from": "gpt", "value": output_text}
    ]

    # 変換後のデータ
    converted_data.append({
        "id": f"conversation_{i+1:04d}",  # `conversation_id` ではなく `id` に統一
        "conversations": conversations    # ShareGPT形式の `conversations` を作成
    })

# JSONファイルとして書き出す
output_file = "merged_sharegpt.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print(f"Converted {len(converted_data)} conversations. Saved to {output_file}")

