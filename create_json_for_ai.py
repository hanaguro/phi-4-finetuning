#!/usr/bin/env python

import json
import os
import requests
from extract_text import extract_text_from_pdf, extract_text_from_mail, is_email_file

# OllamaのAPIエンドポイント
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OUTPUT_DIR = "./output_json"
MODEL = "phi4"
#MODEL = "hf.co/alfredplpl/gemma-2-2b-jpn-it-gguf"
#MODEL = "deepseek-r1:14b"
#MODEL = "gemma2:27b"
#MODEL = "qwen2.5:32b"


def main():
    import argparse
    
    # ArgumentParserのインスタンス生成
    parser = argparse.ArgumentParser(description='Extract json from PDF files.')

    # ディレクトリパス引数の追加（必須、ディレクトリパス）
    parser.add_argument('dir_path', type=str, help='Path to the directory')

    # 引数を解析
    args = parser.parse_args()

    target_files = []
    for root, _, files in os.walk(args.dir_path):
        for file in files:
            target_files.append(os.path.join(root, file))

    for file in target_files:
        save_path = OUTPUT_DIR + os.path.splitext(file)[0] + ".json"
        print(save_path)

        # 既に保存されている場合はスキップ
        if os.path.exists(save_path):
            continue

        extracted_text = ""
        if file.endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file)
        elif is_email_file(file):
            extracted_text = extract_text_from_mail(file)
        else:
            continue

        # リクエストに送信するデータ
        payload = {
            "model": MODEL,
            "prompt": extracted_text + """

            この文字列から読み取れる情報の指示文と出力を作成してください。
            必ずjson形式で作成してください。
            jsonのkeyは必ず"instruction"、"input", "output"のみになります。
            instructionとoutputには主語を必ず入れてください。
            inputには何も入れないでください。
            outputの文字数は多くしてください。多ければ多いほどよいです。
            instructionとoutputだけを読んでも意味が通じるようにしてください。
            instructionとoutput共になるべく詳細に記述してください。
            例えば、「こじまみつひろ氏が中心となって開発しているPlamo Linuxの新版」という文章があったら以下のようになります。
            
            [
            {
            "instruction":"Plamo Linuxの代表者は誰ですか",
            "input":"",
            "output":"こじまみつひろ氏が代表者です。Plamo Linux（プラモ・リナックス）はLinuxディストリビューションの一つです。Slackwareを基本に開発されました。RPMやPortageのようなパッケージシステムを持たず、Slackwareと同じpkgtoolというシンプルなパッケージ管理システムを採用しており、構成がきわめてシンプルになっています。過度なブラックボックス化や自動化を施さず，極力シンプルで見通しの良い環境の保持に努め、Linux の持つボビー性を大切にという考えの基に開発されています。そのため、カスタマイズが非常に簡単にでます。 ",
            }
            ]
        
            他の指示文と出力は文字列を解析して自動で作成してください。
            「本日」「今日」や「この」などを使用されても分からないので、文章から読み取れる日付や「この」などが示す言葉を使用してください。
            「本日」、「今日」などはDate:の後に日付が入っている場合はその日付を使用してください。
            すぐにjson形式で保存できるように、余計な出力はしないでください。
            指示文と出力がなければ空のjsonファイルを作成してください。
            """,
        }
        
        # ヘッダー（必要に応じて追加）
        headers = {
            "Content-Type": "application/json"
        }
        
        # POSTリクエストでAPIを呼び出す
        response = requests.post(OLLAMA_API_URL, json=payload, headers=headers)
        
        # レスポンスの確認と処理
        if response.status_code == 200:
            try:
                # NDJSONを処理
                lines = response.text.strip().split("\n")
                full_response = ""  # 全体のレスポンスを格納する変数
                for line in lines:
                    json_obj = json.loads(line)  # 各行をJSONとしてパース
                    if "response" in json_obj:
                        full_response += json_obj["response"]  # responseキーの値を連結
    
                os.makedirs(os.path.dirname(save_path), exist_ok=True)    

                print(full_response)
                
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(full_response)
    
            except json.JSONDecodeError as e:
                print("JSON Decode Error:", str(e))
                print("Response Text:", response.text)
        else:
            print("HTTPエラー:", response.status_code)
            print("Response Text:", response.text)

if __name__ == "__main__":
    main()
