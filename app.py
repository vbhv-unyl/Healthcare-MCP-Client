import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import agent as agent_module
from api_client import ApiClient

st.set_page_config(page_title="Medical Report Assistant (test)", layout="wide")

_client = ApiClient(base_url=st.secrets["FUNCTION_APP_URL"], function_key=st.secrets["FUNCTION_APP_KEY"])

def _get_or_create_persistent_user_id() -> str:
    user_id = "user1"
    return user_id

if "user_id" not in st.session_state:
    try:
        st.session_state["user_id"] = _get_or_create_persistent_user_id()
    except requests.HTTPError as exc:
        st.error(f"Couldn't start a session: {exc}")
        st.stop()


# =============================================================================
# Sidebar: upload + a session-tracked document list. There's no
# list_documents call anymore (api_client.py was deliberately trimmed to
# just upload_document/get_document_status), so this only ever shows
# documents uploaded THIS session, tracked in memory -- not the full
# history a real account might have. Each one's status is fetched fresh
# on every rerun (including every chat message sent), which is fine for a
# handful of test documents but would be worth replacing with a manual
# refresh button or a cache if this ever needs to hold more than that.
# =============================================================================

if "uploaded_docs" not in st.session_state:
    st.session_state["uploaded_docs"] = []

with st.sidebar:
    st.subheader("Upload a report")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file is not None and st.button("Upload", use_container_width=True):
        try:
            result = _client.upload_document(
                user_id=st.session_state["user_id"],
                filename=uploaded_file.name,
                file_bytes=uploaded_file.getvalue(),
                content_type=uploaded_file.type or "application/pdf",
            )
            st.session_state["uploaded_docs"].append({"doc_id": result["doc_id"], "filename": uploaded_file.name})
            st.success(f"Uploaded — status: {result['status']}")
            st.rerun()
        except requests.HTTPError as exc:
            st.error(f"Upload failed: {exc}")

    st.divider()
    st.subheader("This session's documents")
    if not st.session_state["uploaded_docs"]:
        st.caption("No documents uploaded yet this session.")
    else:
        _STATUS_ICONS = {"pending": "⏳", "processing": "⚙️", "indexed": "✅", "failed": "❌"}
        for doc in st.session_state["uploaded_docs"]:
            try:
                status = _client.get_document_status(st.session_state["user_id"], doc["doc_id"])["status"]
                icon = _STATUS_ICONS.get(status, "•")
                st.write(f"{icon} {doc['filename']} — {status}")
            except requests.HTTPError as exc:
                st.write(f"⚠️ {doc['filename']} — couldn't fetch status ({exc})")


# =============================================================================
# Chat: connects to the MCP server via a LangGraph agent (agent.py). Built
# once per session, cached in session_state. See agent.py's own docstring
# for what's genuinely verified here versus what still needs a real model
# API key to prove end to end, and for the still-open
# user_id-as-tool-parameter gap (mitigated via a header, not eliminated).
# =============================================================================

st.title("Ask about your reports")

if "agent" not in st.session_state:
    with st.spinner("Connecting to the retrieval service..."):
        try:
            st.session_state["agent"] = agent_module.build_agent(user_id=st.session_state["user_id"])
        except Exception as exc:
            st.session_state["agent"] = None
            st.error(f"Couldn't connect to the retrieval service: {exc}")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for role, message in st.session_state["chat_history"]:
    with st.chat_message(role):
        st.write(message)

user_question = st.chat_input("Ask a question about your reports")
if user_question:
    st.session_state["chat_history"].append(("user", user_question))
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        if st.session_state["agent"] is None:
            response = "The retrieval service isn't available right now, so I can't answer that."
            st.write(response)
        else:
            with st.spinner("Thinking..."):
                try:
                    response = agent_module.ask(st.session_state["agent"], user_question)
                except Exception as exc:
                    response = f"Something went wrong answering that: {exc}"
            st.write(response)
    st.session_state["chat_history"].append(("assistant", response))