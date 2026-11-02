import sys
from pathlib import Path
from typing import TypedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from langgraph.graph import StateGraph, END
from chain import CustomerServiceChain


INTENT_DEFINITIONS = """将用户消息分类到以下意图之一:
- 询价: 问安装费、收费标准、价格
- 预约: 问下单、派单、上门时间、指定师傅
- 投诉: 投诉师傅迟到、态度差、没来、安装不满意
- 理赔: 问墙刮花了怎么赔、商品坏了怎么赔、保险
- 退款: 问退款、退费、多久到账
- 咨询: 其他一般性问题

只输出意图名称，不要解释。"""


class AgentState(TypedDict):
    query: str
    intent: str
    answer: str


class IntentGraph:
    def __init__(self):
        self.chain = CustomerServiceChain()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("classify", self._classify_intent)
        builder.add_node("handle_pricing", self._handle_pricing)
        builder.add_node("handle_booking", self._handle_booking)
        builder.add_node("handle_complaint", self._handle_complaint)
        builder.add_node("handle_claim", self._handle_claim)
        builder.add_node("handle_refund", self._handle_refund)
        builder.add_node("handle_general", self._handle_general)

        builder.set_entry_point("classify")

        builder.add_conditional_edges(
            "classify",
            self._route_by_intent,
            {
                "询价": "handle_pricing",
                "预约": "handle_booking",
                "投诉": "handle_complaint",
                "理赔": "handle_claim",
                "退款": "handle_refund",
                "咨询": "handle_general",
            },
        )

        for node in ["handle_pricing", "handle_booking", "handle_complaint",
                      "handle_claim", "handle_refund", "handle_general"]:
            builder.add_edge(node, END)

        return builder.compile()

    def _classify_intent(self, state: AgentState) -> AgentState:
        query = state["query"]
        prompt = f"{INTENT_DEFINITIONS}\n\n消息: {query}"
        try:
            resp = self.chain.llm.invoke([{"role": "user", "content": prompt}])
            intent = resp.content.strip()
            if intent not in ("询价", "预约", "投诉", "理赔", "退款", "咨询"):
                intent = "咨询"
        except Exception:
            intent = "咨询"

        print(f"  [意图] \"{query[:30]}\" → {intent}")
        return {"query": query, "intent": intent, "answer": ""}

    def _route_by_intent(self, state: AgentState) -> str:
        return state["intent"]

    def _handle_pricing(self, state: AgentState) -> AgentState:
        query = state["query"]
        augmented = f"安装费用 收费标准 价格 {query}"
        answer = self.chain.answer(augmented)
        return {"answer": answer}

    def _handle_booking(self, state: AgentState) -> AgentState:
        query = state["query"]
        augmented = f"下单 预约 派单 上门 {query}"
        answer = self.chain.answer(augmented)
        return {"answer": answer}

    def _handle_complaint(self, state: AgentState) -> AgentState:
        query = state["query"]
        augmented = f"投诉 师傅 {query}"
        answer = self.chain.answer(augmented)
        return {"answer": answer}

    def _handle_claim(self, state: AgentState) -> AgentState:
        query = state["query"]
        augmented = f"理赔 赔偿 保险 损坏 {query}"
        answer = self.chain.answer(augmented)
        return {"answer": answer}

    def _handle_refund(self, state: AgentState) -> AgentState:
        query = state["query"]
        augmented = f"退款 退费 到账 {query}"
        answer = self.chain.answer(augmented)
        return {"answer": answer}

    def _handle_general(self, state: AgentState) -> AgentState:
        answer = self.chain.answer(state["query"])
        return {"answer": answer}

    @property
    def last_sources(self) -> list[dict]:
        return getattr(self.chain, "last_sources", [])

    def answer(self, user_input: str) -> str:
        result = self.graph.invoke({"query": user_input, "intent": "", "answer": ""})
        return result["answer"]

    def clear_memory(self):
        self.chain.clear_memory()


_intent_graph: IntentGraph | None = None


def get_intent_graph() -> IntentGraph:
    global _intent_graph
    if _intent_graph is None:
        _intent_graph = IntentGraph()
    return _intent_graph
