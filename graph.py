
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from typing import TypedDict, List, Annotated
import operator
from dotenv import load_dotenv
import os

from tools.rag_tool import rag_search
from tools.web_search_tool import web_search

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    query: str
    final_answer: str
    tool_used: str
    history: List[dict]


class AgentGraph:

    MANAGER_SYSTEM = """You are an intelligent assistant with access to two tools:

1. **rag_search** — Searches internal documents (PDFs, company policies, uploaded knowledge).
   Use when: the query is about internal documentation, uploaded PDFs, or proprietary knowledge.

2. **web_search** — Searches the internet via DuckDuckGo.
   Use when: the query requires current events, recent news, or external public knowledge.

Your behaviour depends on what is in the conversation so far:

CASE A — No tool results yet:
- Decide which tool(s) to call based on the user query.
- Always call at least one tool. Never answer directly in this case.

CASE B — Tool results are present in the message history:
- Read the raw tool results already in the conversation carefully.
- Produce a clear, well-structured, accurate final answer from them.
- Use ONLY what the tools returned. Do not add outside knowledge.
- If a tool returned "NO_RESULTS", ignore it and rely on the other tool's output.
- If ALL tools returned "NO_RESULTS", answer from your own knowledge.
  If you truly cannot answer, say: "I don't have enough information."
- Do NOT call any tool again in this case.

Always be concise, accurate, and helpful. Never hallucinate."""

    def __init__(self):
        self.tools = [rag_search, web_search]
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build_graph()
        self.app = self.graph.compile()

    def manager_node(self, state: AgentState):
        """
        Single node. Single system prompt. Single LLM call per visit.

        Visit 1 (no ToolMessages in state):
            → LLM sees query, picks tools, returns tool_calls  →  routes to tools node.

        Visit 2 (ToolMessages now in state):
            → LLM sees full thread including raw tool output, writes final answer
              (synthesis if useful results exist, own-knowledge fallback if all NO_RESULTS)
            → no tool_calls emitted  →  routes to END.
        """
        history_text = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in state.get("history", [])[-5:]
        )

        existing_messages = state.get("messages", [])
        tool_messages = [m for m in existing_messages if isinstance(m, ToolMessage)]

        if not tool_messages:
            #  Visit 1: build fresh thread, ask manager to pick tools 
            user_content = (
                f"Conversation so far:\n{history_text}\n\nUser query: {state['query']}"
                if history_text
                else state["query"]
            )
            messages = [
                SystemMessage(content=self.MANAGER_SYSTEM),
                HumanMessage(content=user_content),
            ]
        else:
            #  Visit 2: pass the full thread — system prompt's CASE B takes over
            # existing_messages already contains:
            #   [SystemMessage, HumanMessage, AIMessage(tool_calls), ToolMessage, ...]
            messages = existing_messages

        response = self.llm_with_tools.invoke(messages)

        tool_names = [tc["name"] for tc in (response.tool_calls or [])]
        print(f"\n{'='*50}")
        print(f"[MANAGER] visit={'1' if not tool_messages else '2'} | "
              f"tool_calls={tool_names or 'none'}")
        print(f"{'='*50}\n")

        if tool_messages:
            # Visit 2 — set final_answer and tool attribution
            useful = [m for m in tool_messages if m.content != "NO_RESULTS"]
            tool_used = (
                " + ".join(m.name for m in useful) if useful else "dynamic LLM fallback"
            )
            return {
                "messages": [response],
                "final_answer": response.content.strip(),
                "tool_used": tool_used,
            }

        # Visit 1 — just append the response (which carries tool_calls)
        return {"messages": messages + [response]}

    def should_use_tools(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("manager", self.manager_node)
        graph.add_node("tools", self.tool_node)

        graph.set_entry_point("manager")

        graph.add_conditional_edges(
            "manager",
            self.should_use_tools,
            {"tools": "tools", END: END},
        )

        # tools always returns to manager for final answer formatting
        graph.add_edge("tools", "manager")

        return graph


app = AgentGraph().app

