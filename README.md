# 快修到家 AI 客服系统

家居安装服务平台的智能客服系统，基于 RAG（检索增强生成）范式构建。

---

## 技术路线图

```
                          ┌──────────────────────────┐
                          │      用户 (浏览器)         │
                          │   http://localhost:5173   │
                          └────────────┬─────────────┘
                                       │ POST /api/chat
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI :8000 (src/server.py)                    │
│  会话管理: session_id → CustomerServiceChain (TTL 3600s)            │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   对话链 CustomerServiceChain (src/chain.py)         │
│                                                                     │
│  用户输入 ──→ [_rewrite_query] ──→ [retrieve] ──→ [answer]         │
│                                                                     │
│  ┌──────────────────┐                                               │
│  │ 1. Query Rewrite │  带指代词/短追问 → LLM 结合历史改写为完整问题     │
│  │    "超过一小时呢" → "师傅迟到超一小时怎么赔偿"                     │
│  └────────┬─────────┘                                               │
│           ▼                                                         │
│  ┌──────────────────────────────────────────┐                       │
│  │ 2. 双路召回                                │                       │
│  │                                            │                       │
│  │   ┌──────────┐      ┌──────────────┐      │                       │
│  │   │ 向量检索  │      │ BM25 关键词   │      │                       │
│  │   │ BGE-small │      │ rank-bm25    │      │                       │
│  │   │ +ChromaDB│      │ 字符级分词    │      │                       │
│  │   └────┬─────┘      └──────┬───────┘      │                       │
│  │        │    Top-10         │   Top-10      │                       │
│  │        └────────┬──────────┘               │                       │
│  │                 ▼                          │                       │
│  │           合并去重 (按文本去重)              │                       │
│  └─────────────────┬──────────────────────────┘                       │
│                    ▼                                                 │
│  ┌──────────────────────────────────────────┐                       │
│  │ 3. 重排序                                  │                       │
│  │                                            │                       │
│  │   CrossEncoder bge-reranker-base           │                       │
│  │   (query, chunk₁) → 0.97    ┐              │                       │
│  │   (query, chunk₂) → 0.83    ├─→ Top-5      │                       │
│  │   (query, chunk₃) → 0.12    ┘              │                       │
│  │                                            │                       │
│  │   模型不可用时 → RRF 排名融合 (回退)          │                       │
│  └─────────────────┬──────────────────────────┘                       │
│                    ▼                                                 │
│  ┌──────────────────────────────────────────┐                       │
│  │ 4. 阈值过滤                                │                       │
│  │   CrossEncoder 分 < 0 → 丢弃               │                       │
│  │   如果全部丢弃 → 触发防幻觉拒答              │                       │
│  └─────────────────┬──────────────────────────┘                       │
│                    ▼                                                 │
│  ┌──────────────────────────────────────────┐                       │
│  │ 5. Prompt 组装                             │                       │
│  │   ┌─────────────────────────────┐         │                       │
│  │   │ System: 你是小快...防幻觉规则 │         │                       │
│  │   │ System: 知识库参考内容 (Top-5)│         │                       │
│  │   │ System: 对话历史 (Memory×6轮)│         │                       │
│  │   │ User:   原始用户问题          │         │                       │
│  │   └─────────────────────────────┘         │                       │
│  └─────────────────┬──────────────────────────┘                       │
│                    ▼                                                 │
│  ┌──────────────────────────────────────────┐                       │
│  │ 6. DeepSeek API (deepseek-chat)           │                       │
│  │   temperature=0.3  max_tokens=1024        │                       │
│  └─────────────────┬──────────────────────────┘                       │
│                    ▼                                                 │
│              ┌──────────┐                                            │
│              │ 回答 +   │  →  存入 Memory  →  返回前端                │
│              │ 来源引用  │                                            │
│              └──────────┘                                            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│    ChromaDB      │  │  SentenceTrans-  │  │    DeepSeek      │
│    向量库 (本地)  │  │  former (本地)   │  │    API (云端)    │
│   13条知识向量    │  │                  │  │                  │
│  cosine 相似度   │  │  BGE-small-zh    │  │  deepseek-chat   │
│  HNSW 索引       │  │  512维 embedding │  │  OpenAI 兼容接口  │
│                  │  │                  │  │                  │
│                  │  │  CrossEncoder    │  │                  │
│                  │  │  bge-reranker-   │  │                  │
│                  │  │  base (精排)     │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 离线评测管线

```
eval_questions.json (30条)
     │
     ▼
每条: → retrieve → answer ──→ DeepSeek 自动打分
     │
     ├─ Faithfulness      (忠实度)    0.98
     ├─ Answer Relevancy  (回答相关性) 1.00
     └─ Context Precision (检索精度)   0.98

──→ 得分 < 0.7 的条目 → 排查知识库 → 补充/重写知识条目
```

---

## 项目结构

```
kx-aics/
├── data/
│   ├── service_manual.txt          # 服务手册 (类目/收费/师傅管理)
│   ├── complaint_policy.txt        # 投诉理赔规则
│   ├── faq_history.txt             # 客服话术库
│   ├── eval_questions.json         # 评测题库 (30条)
│   └── eval_results.json           # 评测结果
├── src/
│   ├── ingest.py                   # 知识库入库 (切分→向量化→ChromaDB)
│   ├── chain.py                    # 对话链 (改写/召回/精排/生成)
│   ├── server.py                   # FastAPI 后端 (/api/chat, 会话管理)
│   ├── evaluate.py                 # 评测脚本 (LLM-as-Judge)
│   └── app.py                      # CLI 入口
├── frontend/                       # React + Vite 前端
│   └── src/App.jsx                 # 聊天界面组件
├── prompts/system_prompt.txt       # System Prompt 模板
├── run.sh                          # 一键启动/关闭脚本
└── .env                            # DEEPSEEK_API_KEY
```

---

## 技术栈

| 层 | 技术 | 用途 |
|---|---|---|
| 前端 | React + Vite | 聊天界面，热更新 |
| 后端 | FastAPI + Uvicorn | REST API，会话管理 |
| LLM | DeepSeek API (deepseek-chat) | 生成、Query Rewrite、评测打分 |
| 向量检索 | ChromaDB | 语义检索，HNSW 索引 |
| 关键词检索 | BM25 (rank-bm25) | 字面匹配，双路召回 |
| 精排 | CrossEncoder (bge-reranker-base) | 候选重排序 |
| 回退 | RRF 排名融合 | 模型不可用时的备选方案 |
| Embedding | BGE-small-zh (512维) | 向量化 |
| 对话记忆 | LangChain ConversationBufferWindowMemory | 多轮上下文 |
| 评测 | DeepSeek LLM-as-Judge | 忠实度/相关性/精度自动打分 |

---

## 快速启动

```bash
cd /root/projects/kx-aics

# 一键启动前后端（索引缺失或语料有更新时会自动重建）
bash run.sh start

# 浏览器访问
open http://localhost:5173
```

```bash
bash run.sh stop     # 关闭
bash run.sh restart  # 重启
bash run.sh status   # 服务状态 + 索引状态
bash run.sh reindex  # 强制重建知识库索引
python src/evaluate.py  # 跑评测
```

### 关于知识库索引

索引（ChromaDB）存放在 `~/.kx-aics/storage`，**不在仓库里**，`git clone` 后需要构建一次。

`run.sh start` 会自动判断，满足任一条件就重建：

- 索引文件不存在（全新部署）
- `data/` 下有比索引更新的 `.txt` / `.md` 语料

每次启动只多一次 `find` 判断；重建约 30 秒（加载 BGE 模型 + 向量化）。

**手动修改语料后**，`bash run.sh status` 会提示「语料比索引新」，下次 `start` 或执行 `reindex` 即会生效。

---

## 核心特性

- **双路混合召回**: 向量语义 + BM25 关键词并行检索，合并去重
- **CrossEncoder 精排**: 逐对打分，Top-5 精度提升
- **Query Rewriting**: 结合对话历史，将"那谁出？"改写为"退货的运费谁出？"
- **三层防幻觉**: 检索阈值过滤 + Prompt 硬约束 + LLM 自身判断
- **多轮对话记忆**: 6 轮窗口，支持指代消解
- **量化评测体系**: 30 条用例，忠实度 0.98 / 相关性 1.00 / 精度 0.98
