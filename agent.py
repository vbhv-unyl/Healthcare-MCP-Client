"""
MCP client + a hand-rolled LangGraph ReAct agent -- not the prebuilt
create_react_agent helper this replaces, but a StateGraph built by hand:
an "agent" node that calls the LLM, a "tools" node (ToolNode) that
executes whatever tool calls it requested, and a conditional edge routing
between them based on whether the LLM's last message actually contains a
tool call.

Ported from a hand-written example with a few real bugs fixed along the
way, worth naming since they'd otherwise silently break this:
- add_conditional_edges (plural) -- the original called
  add_conditional_edge (singular), which doesn't exist on StateGraph;
  confirmed directly against the installed library, not assumed.
- should_continue's routing map didn't match what the function actually
  returns -- it returned "continue"/"end", but the map's keys were
  "continue"/"exit", so a normal (non-tool-calling) response would have
  hit a routing error on the very first turn that didn't call a tool.
- should_continue's return type annotation said -> AgentState; it
  returns a plain string used to select an edge, never agent state.

Tools are NOT hand-written @tool-decorated functions -- they're loaded
from the MCP server via MultiServerMCPClient, same mechanism the
create_react_agent-based version used. user_id is still bound via an
x-user-id request header, not a tool parameter -- see this project's
server.py for why: a Context-typed parameter is excluded from the tool's
schema entirely (confirmed directly, not assumed), so the model has no
field to put a user_id into, correct or otherwise.

The chat model is Azure OpenAI (AzureChatOpenAI), not the generic
"openai:gpt-4.1" provider string the earlier version used -- matching
the rest of this project, which is Azure-only throughout (Azure OpenAI
embeddings, Azure AI Search, Azure Functions, Azure SQL). This assumes
the SAME Azure OpenAI resource already used for embeddings also hosts a
chat-capable deployment: it reuses AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_KEY
and adds one new env var, AZURE_OPENAI_CHAT_DEPLOYMENT, for the
deployment name specifically. If the chat model actually lives on a
different Azure OpenAI resource, this needs its own separate
endpoint/key vars instead -- not assumed here.

Verified: MCP tool loading, and this exact graph's construction and
routing logic (should_continue correctly selecting "tools" vs "end"),
against a real running MCP server and a mock LLM standing in for
AzureChatOpenAI. NOT verified: the LLM actually reasoning through a
question end to end -- that needs a real Azure OpenAI chat deployment
and API key, neither available in the environment this was built in.
"""

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
    """llm is injectable so tests can supply a mock instead of a real AzureChatOpenAI -- see this file's tests."""
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
                "headers": {"x-user-id": user_id},  # the actual binding -- see this file's module docstring
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