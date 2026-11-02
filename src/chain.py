import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.environ.get(
    "CHROMA_STORAGE",
    str(Path.home() / ".kx-aics" / "storage")
))
PROMPTS_DIR = PROJECT_ROOT / "prompts"

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI


class ModelPool:
    """服务启动时加载一次，所有会话共享"""

    def __init__(self):
        print("[WarmUp] 预加载模型池 ...")

        print("  [1/4] Embedding: BAAI/bge-small-zh-v1.5 ...")
        self.embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        print("  [2/4] ChromaDB ...")
        chroma_client = chromadb.PersistentClient(path=str(STORAGE_DIR))
        self.collection = chroma_client.get_collection("customer_service_knowledge")
        self.documents = self.collection.get(include=["documents"])["documents"]
        print(f"       集合 {self.collection.count()} 条向量")

        print("  [3/4] BM25 索引 ...")
        self.bm25 = BM25Okapi([list(d) for d in self.documents])
        print(f"       文档数 {len(self.documents)}")

        print("  [4/4] Reranker ...")
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("BAAI/bge-reranker-base")
            print(f"       CrossEncoder 就绪")
        except Exception:
            self.reranker = None
            print(f"       RRF 回退模式")

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY")
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=0.3,
            max_tokens=1024,
        )
        print("[WarmUp] 模型池就绪 ✓\n")


_pool: ModelPool | None = None


def get_pool() -> ModelPool:
    global _pool
    if _pool is None:
        _pool = ModelPool()
    return _pool


class CustomerServiceChain:
    """每个会话独立创建，共享底层模型"""

    def __init__(self, pool: ModelPool | None = None):
        self.pool = pool or get_pool()
        self.last_sources: list[dict] = []
        self._init_memory()
        self._init_prompt_template()

    def _init_memory(self):
        self.memory = ConversationBufferWindowMemory(
            k=6, memory_key="chat_history", return_messages=True,
        )

    def _init_prompt_template(self):
        prompt_path = PROMPTS_DIR / "system_prompt.txt"
        if prompt_path.exists():
            self.system_prompt_template = prompt_path.read_text(encoding="utf-8")
        else:
            self.system_prompt_template = (
                '你是一个专业的电商客服助手。请仅基于以下知识库内容回答。\n\n知识库内容:\n{context}'
            )

    def _vector_search(self, query: str, top_k: int) -> list[tuple[float, str]]:
        query_embedding = self.pool.embed_model.encode(query).tolist()
        results = self.pool.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self.pool.documents)),
            include=["documents", "distances"],
        )
        candidates = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                candidates.append((1 - distance, results["documents"][0][i]))
        return candidates

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[float, str]]:
        tokenized_query = list(query)
        scores = self.pool.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(s, self.pool.documents[i]) for i, s in ranked[:top_k] if s > 0]

    def _merge_dedup(self, vec, bm):
        seen = set()
        merged = []
        for score, doc in vec + bm:
            key = doc[:80]
            if key not in seen:
                seen.add(key)
                merged.append((score, doc))
        print(f"  [召回] 向量: {len(vec)} | BM25: {len(bm)} | 合并: {len(merged)}")
        return merged

    def _rerank(self, query, candidates, top_k):
        pairs = [(query, doc) for _, doc in candidates]
        scores = self.pool.reranker.predict(pairs)
        reranked = sorted(
            [(float(s), doc) for s, doc in zip(scores, [d for _, d in candidates])],
            key=lambda x: x[0], reverse=True
        )
        print(f"  [精排] CrossEncoder: {len(candidates)} → Top-{top_k}, "
              f"最高分: {reranked[0][0]:.2f}" if reranked else "  [精排] 无候选")
        return reranked[:top_k]

    def _rrf_fusion(self, vec, bm, top_k, k=60):
        doc_score = {}
        doc_text = {}
        for rank, (score, text) in enumerate(vec):
            key = text[:80]
            doc_text[key] = text
            doc_score[key] = doc_score.get(key, 0) + 1.0 / (k + rank + 1)
        for rank, (score, text) in enumerate(bm):
            key = text[:80]
            doc_text[key] = text
            doc_score[key] = doc_score.get(key, 0) + 1.0 / (k + rank + 1)
        fused = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
        return [(score, doc_text[key]) for key, score in fused[:top_k]]

    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.0):
        pool_size = top_k * 2
        vec = self._vector_search(query, pool_size)
        bm = self._bm25_search(query, pool_size)

        if self.pool.reranker:
            merged = self._merge_dedup(vec, bm)
            final = self._rerank(query, merged, top_k) if len(merged) > top_k else merged
        else:
            final = self._rrf_fusion(vec, bm, top_k)

        return [(s, t) for s, t in final if s >= threshold]

    def _rewrite_query(self, user_input: str) -> str:
        history = self.memory.load_memory_variables({})
        messages = history.get("chat_history", [])
        if not messages:
            return user_input
        if not (any(w in user_input for w in ("那", "这个", "它", "他", "这", "那个"))
                or len(user_input) <= 10):
            return user_input

        history_text = ""
        for msg in messages[-4:]:
            role = "用户" if msg.type == "human" else "客服"
            history_text += f"{role}: {msg.content}\n"

        prompt = (
            "根据对话历史，将模糊追问改写为完整独立的问题。只输出问题。\n\n"
            f"对话历史:\n{history_text}\n用户追问: {user_input}\n改写:"
        )
        try:
            resp = self.pool.llm.invoke([{"role": "user", "content": prompt}])
            rewritten = resp.content.strip()
            if rewritten and rewritten != user_input:
                print(f"  [改写] \"{user_input}\" → \"{rewritten}\"")
                return rewritten
        except Exception:
            pass
        return user_input

    def _build_sources(self, results) -> list[dict]:
        """把检索结果整理成可返回给前端的引用列表"""
        sources = []
        for score, text in results:
            title = "知识库片段"
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    title = line.lstrip("#").strip() or title
                break
            snippet = " ".join(text.split())
            for ch in ("#", "*", "`", "|", ">"):
                snippet = snippet.replace(ch, " ")
            snippet = " ".join(snippet.split())
            sources.append({
                "title": title,
                "content": snippet[:160] + ("…" if len(snippet) > 160 else ""),
                "score": round(float(score), 3),
            })
        return sources

    def answer(self, user_input: str) -> str:
        rewritten = self._rewrite_query(user_input)
        results = self.retrieve(rewritten)
        self.last_sources = self._build_sources(results)

        if not results:
            fallback = (
                "小快：抱歉，根据我目前掌握的服务信息，暂时无法确认这个问题。"
                "建议您拨打客服热线 400-XXX-XXXX 获取更详细的帮助。\n"
                "(系统提示: 知识库中未找到相关度足够高的内容)"
            )
            self._save_to_memory(user_input, fallback)
            return fallback

        contexts = []
        for i, (score, text) in enumerate(results):
            contexts.append(f"【参考资料 {i+1}】(相关度: {score:.2f})\n{text}")
        knowledge = "\n\n---\n\n".join(contexts)

        history = self.memory.load_memory_variables({})
        msgs = history.get("chat_history", [])
        hist_text = ""
        if msgs:
            hist_text = "## 对话历史\n" + "\n".join(
                f"{'用户' if m.type == 'human' else '客服'}: {m.content}" for m in msgs
            )

        prompt = self.system_prompt_template.replace("{context}", knowledge)
        messages = [{"role": "system", "content": prompt}]
        if hist_text:
            messages.append({"role": "system", "content": f"对话历史供参考:\n{hist_text}"})
        messages.append({"role": "user", "content": user_input})

        resp = self.pool.llm.invoke(
            [{"role": m["role"], "content": m["content"]} for m in messages]
        )
        answer = resp.content

        self._save_to_memory(user_input, answer)
        return answer

    def _save_to_memory(self, user_input: str, answer: str):
        self.memory.save_context({"input": user_input}, {"output": answer})

    def clear_memory(self):
        self.memory.clear()
