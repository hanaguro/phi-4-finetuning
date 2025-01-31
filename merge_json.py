#!/usr/bin/env python

import os

import json
import re

def extract_and_parse_json(text: str):
    """
    ファイルのテキストから、最初の '{' か '[' から最後の '}' か ']' までを取り出し、
    必要に応じて '[ ... ]' で囲んで JSON としてパースする。
    """
    # 最初に見つかる '{' または '[' の位置を探す
    start_match = re.search(r'[\{\[]', text)
    if not start_match:
        # JSON らしき部分がなければ空のリストを返す
        print("No { or [ found")
        return []
    start_index = start_match.start()
    
    # 最後に見つかる '}' または ']' の位置を探す
    end_match = re.search(r'[\}\]](?!.*[\}\]])', text)
    if not end_match:
        # JSON らしき部分がなければ空のリストを返す
        print("No } or ] found")
        return []
    end_index = end_match.end()
    
    # 抜き出し
    candidate = text[start_index:end_index].strip()
    
    # もし抜き出した部分が '[' で始まっていなければ、全体を配列として解釈させるために '[...]' で囲む
    if not candidate.startswith('['):
        candidate = f"[{candidate}"

    if not candidate.endswith(']'):
        candidate = f"{candidate}]"
    
    # JSON パースを試みる
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # パースに失敗したら空のリストにしておく (もしくはログ出力など)
        print(f"Failed to parse JSON: {candidate}")
        return []
    
    # パース結果が dict なら単一オブジェクトなのでリストに変換、list ならそのまま返す
    if isinstance(parsed, dict):
        return [parsed]
    elif isinstance(parsed, list):
        return parsed
    else:
        # それ以外の型(文字列や数値など)がトップに来ることは想定外なので空リスト
        print(f"Unexpected type: {type(parsed)}")
        return []

def merge_json_files(directory: str, output_file: str):
    """
    指定ディレクトリ以下の .json ファイルを順に読み込み、抽出・パースした結果を
    1つの JSON 配列にして output_file に出力する。
    """
    all_data = []
    
    # ディレクトリ内のファイルを走査
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".json"):
                continue  # JSON ファイル以外はスキップ
            
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                text = f.read()
            
            # ファイルの内容から JSON を抽出＆パース
            data = extract_and_parse_json(text)
    
            # パース結果を結合
            all_data.extend(data)
    
    # 結合結果を JSON として書き出し
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(all_data, out, ensure_ascii=False, indent=2)


def main():
    import argparse
    
    # ArgumentParserのインスタンス生成
    parser = argparse.ArgumentParser(description='Merge json from JSON files.')

    # ディレクトリパス引数の追加（必須、ディレクトリパス）
    parser.add_argument('dir_path', type=str, help='Path to the directory')

    # 引数を解析
    args = parser.parse_args()

    # 出力先のファイル名
    merged_output = "merged.json"

    merge_json_files(args.dir_path, merged_output)


if __name__ == '__main__':
    main()
