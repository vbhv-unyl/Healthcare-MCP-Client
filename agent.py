import asyncio
import os
import threading
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

_SYSTEM_PROMPT = SystemMessage(
    content="""You help a patient understand their own uploaded medical reports.

Ground every factual claim in a search_report_chunks result -- don't state lab values, \
findings, or dates from memory. If the search results don't cover the question, say so \
plainly rather than guessing.

You are not a doctor. Describe what the reports say; do not diagnose, and do not tell the \
user what to do about a finding beyond suggesting they discuss it with their clinician."""
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _build_llm(tools: list) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        temperature=0,
    ).bind_tools(tools)


def _compile_graph(tools: list, llm=None):
    llm = llm if llm is not None else _build_llm(tools)

    async def model_call(state: AgentState) -> AgentState:
        response = await llm.ainvoke([_SYSTEM_PROMPT, *state["messages"]])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        return "continue" if last_message.tool_calls else "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", model_call)
    graph.add_node("tools", ToolNode(tools=tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ONE persistent event loop, reused for every build_agent()/ask() call --
# NOT asyncio.run(), which creates and destroys a fresh loop on every
# single call. That mismatch is exactly what causes "Event loop is
# closed" on a second call: AzureChatOpenAI's (and the MCP client's)
# internal async HTTP client lazily binds itself to whichever loop is
# running the first time it's actually used, then breaks the next time a
# DIFFERENT, freshly-created loop tries to reuse the same client object --
# confirmed directly by reproducing this exact failure with a minimal
# stand-in before writing this fix, not assumed from memory.
#
# A module-level loop is shared across every Streamlit session in this
# process (Streamlit runs sessions as threads, not separate processes) --
# fine for this single-tester testing setup, but NOT safe for two people
# using the app at the truly same moment (asyncio loops aren't meant to be
# run_until_complete'd concurrently from multiple threads). Revisit this
# if this setup ever needs real concurrent multi-user use.
_loop = asyncio.new_event_loop()
_loop_lock = threading.Lock()

def _run(coro):
    with _loop_lock:
        return _loop.run_until_complete(coro)

async def _build_agent_async(user_id: str):
    client = MultiServerMCPClient(
        {
            "medical_reports": {
                "url": os.environ.get("MCP_SERVER_URL", "http://localhost:8899/mcp"),
                "transport": "streamable_http",
                "headers": {"x-user-id": user_id},
            }
        }
    )
    tools = await client.get_tools()
    return _compile_graph(tools)


def build_agent(user_id: str):
    """Sync wrapper -- runs on the one persistent loop (_loop), not a fresh one, so it shares a loop with every later ask() call."""
    return _run(_build_agent_async(user_id))


async def _ask_async(compiled_graph, question: str) -> str:
    result = await compiled_graph.ainvoke({"messages": [("user", question)]})
    final_message = result["messages"][-1]
    return final_message.content


def ask(compiled_graph, question: str) -> str:
    return _run(_ask_async(compiled_graph, question))