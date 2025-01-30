#!/usr/bin/env python

import pymupdf4llm
from email.parser import BytesParser
from email.policy import default

def extract_text_from_pdf(pdf_path):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    md_text = md_text.replace("\n\n", "\n")

    return md_text


def is_email_file(file_path):
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()

        # メールのパースを試みる
        message = BytesParser(policy=default).parsebytes(raw_data)

        # ヘッダの基本的な要素を確認
        if message["From"] and message["To"] and message["Subject"]:
            return True
        else:
            return False
    except Exception:
        return False  # 何らかのエラーが発生した場合はメールでないと判断


def extract_text_from_mail(mail_path):
    with open(mail_path, "rb") as f:
        raw_data = f.read()

    message = BytesParser(policy=default).parsebytes(raw_data)

    subject = message["Subject"] or "(No Subject)"

    body_part = message.get_body(preferencelist=("plain", "html"))
    body = body_part.get_content() if body_part else "(No Body)"

    text = "\n\n".join(filter(None, [subject, body]))

    return text


def main():
    import argparse
    
    # ArgumentParserのインスタンス生成
    parser = argparse.ArgumentParser(description='Extract text from a PDF file.')

    # ファイルパス引数の追加（必須、ファイルパス）
    parser.add_argument('file_path', type=str, help='Path to the file')

    # 引数を解析
    args = parser.parse_args()

    extracted_text = ""
    if args.file_path.endswith(".pdf"):
        # PDFファイルの場合
        extracted_text = extract_text_from_pdf(args.file_path)
    else:
        # メールファイルの場合
        extracted_text = extract_text_from_mail(args.file_path)

    if extracted_text:
        print(extracted_text)


if __name__ == "__main__":
    main()


