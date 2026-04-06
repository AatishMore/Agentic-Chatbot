
import streamlit as st
from spellchecker import SpellChecker
import os
import requests
import json

FASTAPI_BASE_URL = "http://localhost:8000"  

class ChatbotUI:
    BADGE_MAP = {
        "rag_search":               "📚 Internal Docs (RAG)",
        "web_search":               "🌐 Web Search (DuckDuckGo)",
        "rag_search + web_search":  "📚 RAG + 🌐 Web (combined)",
        "web_search + rag_search":  "🌐 Web + 📚 RAG (combined)",
        "dynamic LLM fallback":     "🧠 Direct LLM",
        "none":                     "⚠️ Tool failed",
    }

    CSS = """
<style>
body { background: linear-gradient(135deg, #0f2027, #2c5364); }
.main-title {
    text-align: center; font-size: 2.5rem;
    font-weight: bold; color: white; animation: fadeIn 1s ease-in-out;
}
.subtitle {
    text-align: center; color: #dcdcdc;
    margin-bottom: 25px; animation: fadeIn 1.5s ease-in-out;
}
.glass {
    background: rgba(255,255,255,0.1); backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px); border-radius: 20px; padding: 10px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2); animation: fadeInUp 0.4s ease;
}
.chat-bubble-user {
    background: rgba(76,175,80,0.85);
    padding: 10px 15px; border-radius: 15px; color: white;
}
.chat-bubble-bot {
    background: rgba(255,255,255,0.9);
    padding: 10px 15px; border-radius: 15px; color: black;
}
.badge { font-size: 0.75rem; opacity: 0.8; margin-top: 4px; }
@keyframes fadeIn   { from { opacity: 0; } to { opacity: 1; } }
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0);    }
}
</style>
"""

    def __init__(self):
        self.spell = SpellChecker()
        self._setup_page()

    def _setup_page(self):
        st.set_page_config(page_title="LangGraph Chatbot", layout="centered")
        st.markdown(self.CSS, unsafe_allow_html=True)
        st.markdown('<div class="main-title">🤖 LangGraph Agentic Chatbot</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">RAG + Web Search — powered by LangGraph & LLM Tool Binding</div>',
            unsafe_allow_html=True
        )

        if "history" not in st.session_state:
            st.session_state.history = []

        if "uploaded_files" not in st.session_state:
            try:
                resp = requests.get(f"{FASTAPI_BASE_URL}/indexed_files", timeout=5)
                st.session_state.uploaded_files = set(resp.json().keys()) if resp.ok else set()
            except Exception:
                st.session_state.uploaded_files = set()

    def _render_upload_section(self):
        st.subheader("📄 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
        )

        if uploaded_files:
            for file in uploaded_files:
                import hashlib
                content_hash = hashlib.sha256(file.getvalue()).hexdigest()

                if content_hash not in st.session_state.uploaded_files:
                    with st.spinner(f"📚 Processing {file.name}..."):
                        self._handle_pdf_upload(file)
                    st.session_state.uploaded_files.add(content_hash)
                else:
                    st.info(f"⏭️ '{file.name}' already indexed — skipping.")

    def _handle_pdf_upload(self, uploaded_file):
        """Send PDF to FastAPI /upload_pdf endpoint."""
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            resp = requests.post(f"{FASTAPI_BASE_URL}/upload_pdf", files=files, timeout=60)
            data = resp.json()

            if data.get("status") == "success":
                st.success(f" {data['message']}")
            elif data.get("status") == "skipped":
                st.info(f" {data['message']}")
            else:
                st.error(f"❌ {data.get('message', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach FastAPI server. Is it running on port 8000?")
        except Exception as e:
            st.error(f"❌ Failed to upload {uploaded_file.name}: {e}")

    def _preprocess_input(self, user_input: str) -> str:
        user_input = user_input.strip() or "Please provide a valid question."
        misspelled = self.spell.unknown(user_input.split())
        if misspelled:
            user_input += f" (Note: possible misspellings: {', '.join(misspelled)})"
        return user_input

    def _safe_run_agent(self, user_input: str, history: list) -> dict:
        """Call FastAPI /chat endpoint instead of running agent directly."""
        try:
            resp = requests.post(
                f"{FASTAPI_BASE_URL}/chat",
                data={
                    "query": user_input,
                    "history": json.dumps(history)
                },
                timeout=60
            )
            if resp.ok:
                return resp.json()
            else:
                return {
                    "final_answer": f"⚠️ Server error ({resp.status_code}). Please try again.",
                    "tool_used": "none"
                }
        except requests.exceptions.ConnectionError:
            return {
                "final_answer": " Cannot reach FastAPI server. Is it running on port 8000?",
                "tool_used": "none"
            }
        except Exception as e:
            print(f"Agent error: {e}")
            return {
                "final_answer": "⚠️ Sorry, there was an issue. Please try rephrasing.",
                "tool_used": "none"
            }

    # ── Chat Rendering ───────────────────────────────────────────────────────
    def _render_history(self):
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                cls = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
                st.markdown(
                    f'<div class="glass"><div class="{cls}">{msg["content"]}</div></div>',
                    unsafe_allow_html=True
                )
                if msg.get("badge"):
                    st.markdown(
                        f'<div class="badge">{msg["badge"]}</div>',
                        unsafe_allow_html=True
                    )

    def _handle_user_input(self, user_input: str):
        user_input = self._preprocess_input(user_input)
        st.session_state.history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(
                f'<div class="glass"><div class="chat-bubble-user">{user_input}</div></div>',
                unsafe_allow_html=True
            )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = self._safe_run_agent(user_input, st.session_state.history)

            answer    = result.get("final_answer", "Sorry, something went wrong.")
            tool_used = result.get("tool_used", "unknown")
            badge     = self.BADGE_MAP.get(tool_used, f"🔧 {tool_used}")

            st.markdown(
                f'<div class="glass"><div class="chat-bubble-bot">{answer}</div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="badge">Tool used: <b>{badge}</b></div>',
                unsafe_allow_html=True
            )

        st.session_state.history.append({
            "role":    "assistant",
            "content": answer,
            "badge":   f"Tool used: **{badge}**"
        })

    # ── Entry Point ──────────────────────────────────────────────────────────
    def run(self):
        self._render_upload_section()
        self._render_history()
        user_input = st.chat_input("💬 Ask anything...")
        if user_input:
            self._handle_user_input(user_input)


ui = ChatbotUI()
ui.run()