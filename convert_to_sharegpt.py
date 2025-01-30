import json

# Alpaca形式のJSONファイルを読み込む
with open("merged.json", "r", encoding="utf-8") as f:
    alpaca_data = json.load(f)

converted_data = []
for i, item in enumerate(alpaca_data):
    # "instruction", "input", "output" がある想定
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output_text = item.get("output", "")

    # ユーザのメッセージ: instruction + (改行してinputを続ける) などお好みで
    user_message = instruction
    if input_text:
        # 必要に応じて結合方式を変える
        user_message += f"\n{input_text}"

    # それぞれ ShareGPT形式の会話オブジェクトに
    conversations = [
        {"from": "human", "value": user_message},
        {"from": "assistant", "value": output_text},
    ]

    converted_data.append({
        "id": f"conversation_{i+1:04d}",
        "conversations": conversations
    })

# ShareGPT形式のJSONファイルとして書き出す
with open("merged_sharegpt.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

