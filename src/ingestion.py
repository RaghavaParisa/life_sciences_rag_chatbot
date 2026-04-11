import os
import json
import pandas as pd
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(data_dir):
    documents = []
    file_map = {}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if not os.path.isfile(path):
            continue

        ext = file.lower().split(".")[-1]
        print(f"Processing {ext.upper()}: {file}")

        try:
            if ext == "csv":
                df = pd.read_csv(path)
                df = df.fillna("")

                for _, row in df.iterrows():
                    content = " | ".join(
                        [f"{col}: {row[col]}" for col in df.columns if str(row[col]).strip()]
                    )
                    if content.strip():
                        chunks = splitter.split_text(content)
                        for chunk in chunks:
                            documents.append({
                                "content": chunk,
                                "source": file,
                                "page": None
                            })

            elif ext in ["xlsx", "xls"]:
                df = pd.read_excel(path)
                df = df.fillna("")

                for _, row in df.iterrows():
                    content = " | ".join(
                        [f"{col}: {row[col]}" for col in df.columns if str(row[col]).strip()]
                    )
                    if content.strip():
                        chunks = splitter.split_text(content)
                        for chunk in chunks:
                            documents.append({
                                "content": chunk,
                                "source": file,
                                "page": None
                            })

            elif ext == "pdf":
                reader = PdfReader(path)
                full_text = ""

                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

                chunks = splitter.split_text(full_text)
                for chunk in chunks:
                    documents.append({
                        "content": chunk,
                        "source": file,
                        "page": None  # Could track page if needed
                    })

            elif ext in ["txt", "md", "json"]:
                with open(path, "r", encoding="utf-8") as f:
                    if ext == "json":
                        data = json.load(f)
                        content = json.dumps(data, indent=2)
                    else:
                        content = f.read()

                chunks = splitter.split_text(content)
                for chunk in chunks:
                    documents.append({
                        "content": chunk,
                        "source": file,
                        "page": None
                    })

            else:
                print(f"Skipped unsupported file type: {file}")
                continue

            # Track file changes
            file_map[file] = os.path.getmtime(path)

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    # Summary
    print(f"\nTotal documents: {len(documents)}")
    print("\nDocument Distribution:")
    source_count = {}
    for doc in documents:
        src = doc.get("source")
        source_count[src] = source_count.get(src, 0) + 1

    for k, v in source_count.items():
        print(f"{k} -> {v} chunks")

    return documents, file_map