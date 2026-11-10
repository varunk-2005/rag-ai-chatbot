"""
Agentic RAG Chatbot with Model Context Protocol (MCP) Support.
Built with Streamlit, Google Gemini, and the Model Context Protocol (MCP).
Runs locally or on Streamlit Community Cloud.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# Internal modular imports
from rag_engine import extract_text_from_pdfs, chunk_documents, RAGVectorStore
from mcp_client import MCPServerManager
from agent import RAGAgent

# Load local environment variables if present
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic RAG + MCP Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished interface
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E88E5;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #718096;
        margin-bottom: 1.5rem;
    }
    .badge-mcp {
        background-color: #4A148C;
        color: #E1BEE7;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 6px;
    }
    .badge-rag {
        background-color: #0D47A1;
        color: #BBDEFB;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 6px;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_stats" not in st.session_state:
    st.session_state.document_stats = {}

if "mcp_manager" not in st.session_state:
    manager = MCPServerManager()
    # Auto-register built-in MCP server
    builtin_server_path = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    if os.path.exists(builtin_server_path):
        manager.register_server("builtin_tools", sys.executable, [builtin_server_path])
    st.session_state.mcp_manager = manager


# --- Helper: API Key Discovery ---
def get_api_key():
    # 1. User manual input from sidebar
    if st.session_state.get("user_api_key"):
        return st.session_state.user_api_key.strip()
    # 2. Streamlit Cloud secrets
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    # 3. Environment variable (.env or system)
    return os.getenv("GOOGLE_API_KEY", "")


api_key = get_api_key()


# --- Sidebar Navigation & Controls ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # 1. API Configuration
    st.subheader("🔑 Authentication")
    if not api_key:
        st.warning("Google Gemini API key required.")
        input_key = st.text_input("Enter Gemini API Key", type="password", help="Get your API key at https://aistudio.google.com/")
        if input_key:
            st.session_state.user_api_key = input_key
            st.rerun()
    else:
        st.success("API Key Active 🟢")
        with st.expander("Change API Key"):
            new_key = st.text_input("New API Key", type="password")
            if st.button("Update Key"):
                st.session_state.user_api_key = new_key
                st.rerun()

    # 2. Model Settings
    st.subheader("🧠 Model Settings")
    selected_model = st.selectbox(
        "Gemini Model",
        options=["gemini-3.6-flash"],
        index=0,
        help="Model used by the agent for reasoning and tool calls."
    )

    # Embeddings are computed with the Gemini embedding API (see rag_engine.py)
    provider_key = "gemini"
    st.caption("Embeddings: Gemini embedding API")

    st.divider()

    # 3. PDF Document Management
    st.subheader("📄 Document Ingestion (RAG)")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents to build the agent knowledge base."
    )

    if st.button("🚀 Process & Build Index", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please provide a Gemini API key first.")
        elif not uploaded_files:
            st.warning("Please select at least one PDF file.")
        else:
            with st.spinner("Extracting text and chunking documents..."):
                try:
                    pages = extract_text_from_pdfs(uploaded_files)
                    if not pages:
                        st.error("No extractable text found in uploaded PDFs.")
                    else:
                        chunks = chunk_documents(pages)
                        progress_bar = st.progress(0, text="Initializing Vector Store...")

                        def update_progress(ratio, text):
                            progress_bar.progress(int(ratio * 100), text=text)

                        store = RAGVectorStore(api_key=api_key, embedding_provider=provider_key)
                        store.build_index(chunks, progress_callback=update_progress)
                        
                        st.session_state.vector_store = store
                        st.session_state.document_stats = store.document_stats
                        progress_bar.empty()
                        st.success(f"Indexed {len(chunks)} chunks across {len(uploaded_files)} PDF(s)!")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")

    # Show Document Stats if indexed
    if st.session_state.vector_store and st.session_state.document_stats:
        stats = st.session_state.document_stats
        with st.expander("📊 Document Statistics", expanded=False):
            st.write(f"**Total Documents:** {stats.get('total_documents', 0)}")
            st.write(f"**Total Chunks:** {stats.get('total_chunks', 0)}")
            st.write(f"**Total Words:** ~{stats.get('total_words', 0):,}")
            st.write("**Files:**")
            for doc in stats.get("documents", []):
                st.write(f"- `{doc}`")

    st.divider()

    # 4. Model Context Protocol (MCP) Tools
    st.subheader("🔌 Model Context Protocol (MCP)")
    mcp_manager: MCPServerManager = st.session_state.mcp_manager
    connected_tools = mcp_manager.get_all_tools()
    
    st.markdown(f"**Connected MCP Servers:** `{len(mcp_manager.clients)}`")
    st.markdown(f"**Active MCP Tools:** `{len(connected_tools)}`")
    
    with st.expander("Inspect Available MCP Tools", expanded=False):
        for tool in connected_tools:
            st.markdown(f"🔧 **`{tool.get('name')}`**")
            st.caption(tool.get('description', 'No description'))

    st.divider()

    # 5. Conversation Controls & Export
    st.subheader("💾 Chat Options")
    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if col2.button("🔄 Reset All", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.document_stats = {}
        st.rerun()

    if st.session_state.messages:
        export_data = []
        for m in st.session_state.messages:
            export_data.append({
                "Role": m["role"],
                "Content": m["content"],
                "Timestamp": m.get("timestamp", datetime.now().isoformat())
            })
        df_export = pd.DataFrame(export_data)
        csv_bytes = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Chat (CSV)",
            data=csv_bytes,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# --- Main Chat Interface ---
st.markdown('<div class="main-title">🤖 Agentic RAG Assistant <span class="badge-rag">RAG</span><span class="badge-mcp">MCP</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent AI Agent with Multi-Step Reasoning, PDF Document Retrieval, and Model Context Protocol (MCP) Extensibility.</div>', unsafe_allow_html=True)

# Show helpful onboarding banner if no document is loaded
if not st.session_state.vector_store:
    st.info("💡 **Getting Started:** Upload your PDF documents in the sidebar and click **Process & Build Index**. Once ready, you can chat with the Agent. The Agent can also access external tools via Model Context Protocol (MCP)!")

# Render Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # If this assistant message contains reasoning steps, show them in an expander
        if msg["role"] == "assistant" and msg.get("steps"):
            with st.expander("🔍 View Agent Reasoning & Tool Steps", expanded=False):
                for step in msg["steps"]:
                    st_type = step.get("step_type")
                    if st_type == "thought":
                        st.markdown(f"🧠 **Thought:** {step.get('content')}")
                    elif st_type == "tool_call":
                        st.markdown(f"🛠️ **Action:** `{step.get('tool_name')}` with arguments: `{json.dumps(step.get('tool_args', {}))}`")
                    elif st_type == "observation":
                        st.markdown(f"📋 **Observation:**\n```\n{step.get('content')[:500]}...\n```")
        st.markdown(msg["content"])


# --- Chat Input & Agent Execution Loop ---
user_query = st.chat_input("Ask a question about your documents, request calculations, or trigger MCP tools...")

if user_query:
    if not api_key:
        st.error("Please provide your Google Gemini API Key in the sidebar to chat.")
    else:
        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        with st.chat_message("user"):
            st.markdown(user_query)

        # Agent Reasoning & Response
        with st.chat_message("assistant"):
            agent = RAGAgent(
                api_key=api_key,
                vector_store=st.session_state.vector_store,
                mcp_manager=st.session_state.mcp_manager,
                model_name=selected_model
            )

            status_container = st.status("🤖 Agent is thinking and planning...", expanded=True)
            step_records = []
            final_response = ""

            try:
                agent_generator = agent.run_agent_loop(
                    user_message=user_query,
                    # Exclude the message just appended, so the current
                    # question is not sent twice (history + request)
                    conversation_history=st.session_state.messages[:-1]
                )

                for step in agent_generator:
                    step_records.append(step.to_dict())
                    
                    if step.step_type == "thought":
                        status_container.write(f"🧠 **Thought:** {step.content}")
                    elif step.step_type == "tool_call":
                        status_container.write(f"🛠️ **Executing Tool:** `{step.tool_name}` with `{json.dumps(step.tool_args)}`")
                    elif step.step_type == "observation":
                        preview = (step.content[:180] + "...") if len(step.content) > 180 else step.content
                        status_container.write(f"📋 **Result:** {preview}")
                    elif step.step_type == "final":
                        final_response = step.content

                status_container.update(label="✅ Agent execution complete!", state="complete", expanded=False)

            except Exception as e:
                status_container.update(label="⚠️ Error during execution", state="error", expanded=False)
                final_response = f"An error occurred while processing your request: {str(e)}"

            # Render final answer
            st.markdown(final_response)

            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
                "steps": step_records,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
