import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = Path(os.environ.get(
    "CHROMA_STORAGE",
    str(Path.home() / ".kx-aics" / "storage")
))

import chromadb
from sentence_transformers import SentenceTransformer


def load_documents(data_dir: Path):
    if not data_dir.exists():
        print(f"[错误] 数据目录不存在: {data_dir}")
        sys.exit(1)

    files = list(data_dir.glob("*"))
    txt_files = [f for f in files if f.suffix.lower() in (".txt", ".md")]
    if not txt_files:
        print(f"[错误] data/ 目录下没有 txt/md 文件")
        sys.exit(1)

    print(f"[加载] 找到 {len(txt_files)} 个文档:")
    chunks = []
    for f in txt_files:
        print(f"  - {f.name}")
        text = f.read_text(encoding="utf-8")
        paragraphs = text.split("\n\n")
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current_chunk) + len(para) > 512:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    print(f"[加载] 共切分为 {len(chunks)} 个文本块")
    return chunks


def build_index(chunks, storage_dir: Path):
    storage_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Embedding] 加载模型: BAAI/bge-small-zh-v1.5 ...")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    print(f"[Embedding] 模型加载完成")

    print(f"[Embedding] 向量化 {len(chunks)} 个文本块...")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=8)

    print(f"[ChromaDB] 初始化 (路径: {storage_dir}) ...")
    client = chromadb.PersistentClient(path=str(storage_dir))
    collection_name = "customer_service_knowledge"

    try:
        client.delete_collection(collection_name)
        print(f"[ChromaDB] 已删除旧集合")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks,
    )

    print(f"[ChromaDB] 写入完成: {len(chunks)} 条向量")
    print(f"[完成] 知识库已就绪，存储于: {storage_dir}")


def main():
    print("=" * 50)
    print("  智能问答客服 - 知识库索引构建")
    print("=" * 50)

    chunks = load_documents(DATA_DIR)
    build_index(chunks, STORAGE_DIR)


if __name__ == "__main__":
    main()
