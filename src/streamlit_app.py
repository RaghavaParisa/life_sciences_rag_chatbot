from sentence_transformers import SentenceTransformer
import streamlit as st
import os
import tempfile
import time
import requests
import json

from auth import authenticate, verify_token
from embeddings import MODEL_PATH, load_or_create_faiss
from ingestion import load_documents
from rag import init_hybrid, retrieve, generate_answer


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Life Sciences RAG",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------
# HIDE SIDEBAR BEFORE LOGIN
# -----------------------------
if "token" not in st.session_state or not st.session_state.token:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

if "app_ready" not in st.session_state:
    st.session_state.app_ready = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

# -----------------------------
# CACHE RAG INIT
# -----------------------------
@st.cache_resource
def init_rag_once():
    index, documents = load_or_create_faiss(DATA_DIR)
    init_hybrid(documents, index)
    return True
@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "all-MiniLM-L6-v2")
    return SentenceTransformer(MODEL_PATH)
    # return SentenceTransformer("models/all-MiniLM-L6-v2")

# -----------------------------
# SESSION STATE
# -----------------------------
if "token" not in st.session_state:
    st.session_state.token = None

if "role" not in st.session_state:
    st.session_state.role = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "custom_docs" not in st.session_state:
    st.session_state.custom_docs = False


# -----------------------------
# LLM AS JUDGE
# -----------------------------
def llm_judge(question, answer):
    prompt = f"""
You are a strict Life Sciences RAG assistant.

Rules:
- Answer ONLY from provided context
- If answer not present → say "Not found in context"
- Do NOT hallucinate
- Be concise and accurate

STRICT RULES:
- Return ONLY valid JSON
- No explanation, no extra text
- Values must be between 0 and 1

Format:
{{
  "faithfulness": float,
  "relevance": float,
  "correctness": float,
  "final_score": float,
  "verdict": "good" or "partial" or "poor"
}}

Question: {question}
Answer: {answer}
"""

    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            },
            timeout=60
        )

        raw_output = res.json().get("response", "")

        # 🔥 DEBUG (IMPORTANT)
        print("\nLLM RAW OUTPUT:\n", raw_output)

        # ✅ Extract JSON block safely
        import re
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)

        if match:
            json_str = match.group(0)
            return json.loads(json_str)

        else:
            print("❌ No JSON found in output")
            return default_judge()

    except Exception as e:
        print("❌ Judge Error:", e)
        return default_judge()


def default_judge():
    return {
        "faithfulness": 0.0,
        "relevance": 0.0,
        "correctness": 0.0,
        "final_score": 0.0,
        "verdict": "error"
    }


# -----------------------------
# LOGIN PAGE
# -----------------------------
def login_page():
    st.title("🔐 Login")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if "logging_in" not in st.session_state:
            st.session_state.logging_in = False

        login_clicked = st.button("Login", disabled=st.session_state.logging_in)

        if login_clicked:
            st.session_state.logging_in = True
            st.session_state.username_tmp = username
            st.session_state.password_tmp = password
            st.rerun()

    # 🔥 AUTH FLOW (runs AFTER rerun → spinner visible)
    if st.session_state.get("logging_in", False):
        with col2:
            with st.spinner("🔐 Authenticating..."):
                time.sleep(0.3)

                success, role, token = authenticate(
                    st.session_state.username_tmp,
                    st.session_state.password_tmp
                )

                if success:
                    st.session_state.token = token
                    st.session_state.role = role
                    st.session_state.logging_in = False

                    st.success("✅ Login successful")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.session_state.logging_in = False
                    st.error("❌ Invalid credentials")

# def login_page():
#     st.title("🔐 Login")

#     col1, col2, col3 = st.columns([1, 2, 1])

#     with col2:
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")

#         if "logging_in" not in st.session_state:
#             st.session_state.logging_in = False

#         login_clicked = st.button(
#             "Login",
#             disabled=st.session_state.logging_in
#         )

#         if login_clicked:
#             st.session_state.logging_in = True

#             # 🔥 Loader
#             with st.spinner("🔐 Authenticating... Please wait"):
#                 success, role, token = authenticate(username, password)

#             # ✅ 👉 PLACE YOUR BLOCK HERE
#             if success:
#                 st.session_state.token = token
#                 st.session_state.role = role
#                 st.session_state.logging_in = False

#                 # st.success("✅ Login successful")
#                 st.toast("Login successful!", icon="✅")

#                 time.sleep(0.3)  # smooth transition
#                 st.rerun()

#             else:
#                 st.session_state.logging_in = False
#                 st.error("❌ Invalid credentials")

    # -----------------------------
    # PHASE 2 → SHOW SPINNER
    # -----------------------------
    if st.session_state.logging_in:
        with col2:
            with st.spinner("🔐 Authenticating... Please wait"):
                time.sleep(0.5)  # ensures spinner renders

                success, role, token = authenticate(
                    st.session_state.username_tmp,
                    st.session_state.password_tmp
                )

                if success:
                    st.session_state.token = token
                    st.session_state.role = role
                    st.session_state.logging_in = False

                    st.success("✅ Login successful")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.session_state.logging_in = False
                    st.error("❌ Invalid credentials")
# -----------------------------
# SIDEBAR
# -----------------------------
# def sidebar():
#     st.sidebar.title("🧬 Dashboard")
#     st.sidebar.markdown(f"**Role:** {st.session_state.role}")


    # if st.sidebar.button("Logout"):
    #     st.session_state.token = None
    #     st.session_state.role = None
    #     st.session_state.chat_history = []
    #     st.session_state.custom_docs = False
    #     st.rerun()


# -----------------------------
# FILE UPLOAD
# -----------------------------
def upload_section():
    st.subheader("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        temp_dir = tempfile.mkdtemp()

        for file in uploaded_files:
            path = os.path.join(temp_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.read())

        st.success("Files uploaded")

        documents, _ = load_documents(temp_dir)
        st.session_state.documents = documents
        init_hybrid(documents, index=None)
        st.session_state.custom_docs = True

        st.session_state.custom_docs = True

        st.success("✅ Instant RAG ready!")

def chat_section():
    st.subheader("💬 Ask Questions")

    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    if "current_query" not in st.session_state:
        st.session_state.current_query = ""

    # -----------------------------
    # INPUT FIRST (ChatGPT style)
    # -----------------------------
    query = st.text_input(
        "🔍 Ask a life sciences question...",
        placeholder="e.g. What is MEPS dataset?",
        disabled=st.session_state.is_processing
    )

    ask_clicked = st.button(
        "Ask",
        disabled=st.session_state.is_processing
    )

    if ask_clicked:
        if not query.strip():
            st.warning("Enter a question")
            return

        st.session_state.is_processing = True
        st.session_state.current_query = query
        st.rerun()

    # -----------------------------
    # PROCESSING (NO REPLACEMENT)
    # -----------------------------
    if st.session_state.is_processing:
        query = st.session_state.current_query

        st.markdown(f"🧑 {query}")
        thinking = st.empty()
        thinking.markdown("🤖 *AI is thinking...*")

        start = time.time()

        try:
            # Init RAG
            if "documents" not in st.session_state:
                index, documents = load_or_create_faiss(DATA_DIR)
                st.session_state.documents = documents
                init_hybrid(documents, index)
            else:
                if st.session_state.custom_docs:
                    init_hybrid(st.session_state.documents, index=None)
                else:
                    index, _ = load_or_create_faiss(DATA_DIR)
                    init_hybrid(st.session_state.documents, index)

            contexts, citations, _ = retrieve(query)

            # ✅ Deduplicate sources
            unique_sources = list(dict.fromkeys(citations))

            # ✅ Clean answer properly
            import re
            raw_answer = generate_answer(query, contexts, citations)
            answer = re.split(r"Sources?:", raw_answer, flags=re.IGNORECASE)[0].strip()

            thinking.empty()

            # -----------------------------
            # STREAM RESPONSE (FINAL UI)
            # -----------------------------
            answer_placeholder = st.empty()
            full = ""

            for word in answer.split():
                full += word + " "
                answer_placeholder.markdown(
                    f"""
                    <div style='background:#ffffff;padding:12px;border-radius:10px;border:1px solid #e0e0e0;color:#000;'>
                    🤖 {full}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                time.sleep(0.02)

            # -----------------------------
            # SOURCES
            # -----------------------------
            if unique_sources:
                with st.expander("📚 Sources"):
                    for s in unique_sources:
                        st.markdown(f"- {s}")

            # -----------------------------
            # EVALUATION
            # -----------------------------
            judge = llm_judge(query, answer)
            latency = round(time.time() - start, 2)

            # ✅ Save ONCE
            st.session_state.chat_history.insert(0, {
                "query": query,
                "answer": answer,
                "latency": latency,
                "judge": judge,
                "sources": unique_sources
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

        st.session_state.is_processing = False
        st.session_state.current_query = ""
        st.rerun()

    # -----------------------------
    # CHAT DISPLAY (Newest on top)
    # -----------------------------
    for chat in st.session_state.chat_history:

        st.markdown(
            f"""
            <div style='background:#f1f3f4;padding:12px;border-radius:10px;margin-bottom:6px;color:#000;'>
            🧑 {chat['query']}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style='background:#ffffff;padding:12px;border-radius:10px;margin-bottom:10px;border:1px solid #e0e0e0;color:#000;'>
            🤖 {chat['answer']}
            </div>
            """,
            unsafe_allow_html=True
        )

        if chat.get("sources"):
            with st.expander("📚 Sources"):
                for s in chat["sources"]:
                    st.markdown(f"- {s}")

        st.caption(f"⏱ {chat['latency']} sec")

        judge = chat.get("judge", {})
        if judge:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Faithfulness", round(judge.get("faithfulness", 0), 2))
            col2.metric("Relevance", round(judge.get("relevance", 0), 2))
            col3.metric("Correctness", round(judge.get("correctness", 0), 2))
            col4.metric("Score", round(judge.get("final_score", 0), 2))

        st.markdown("---")
# def chat_section():
#     st.subheader("💬 Ask Questions")

#     # -----------------------------
#     # SESSION FLAGS
#     # -----------------------------
#     if "is_processing" not in st.session_state:
#         st.session_state.is_processing = False

#     if "current_query" not in st.session_state:
#         st.session_state.current_query = ""

#     # -----------------------------
#     # INPUT
#     # -----------------------------
#     query = st.text_input(
#         "🔍 Ask a life sciences question...",
#         placeholder="e.g. What is MEPS dataset?",
#         disabled=st.session_state.is_processing  # 🔥 disable input
#     )

#     # -----------------------------
#     # BUTTON
#     # -----------------------------
#     ask_clicked = st.button(
#         "Ask",
#         disabled=st.session_state.is_processing  # 🔥 disable button
#     )

#     # -----------------------------
#     # TRIGGER PROCESS
#     # -----------------------------
#     if ask_clicked:
#         if not query.strip():
#             st.warning("Enter a question")
#             return

#         # 🔥 Lock UI instantly
#         st.session_state.is_processing = True
#         st.session_state.current_query = query
#         st.rerun()

#     # -----------------------------
#     # PROCESSING BLOCK
#     # -----------------------------
#     if st.session_state.is_processing:
#         query = st.session_state.current_query

#         # 🔥 Show UI immediately
#         status_box = st.empty()

#         with status_box.container():
#             with st.status("🔄 Processing your query...", expanded=True) as status:
#                 progress = st.progress(0)

#                 start = time.time()

#                 try:
#                     # ✅ Ensure hybrid init
#                     if "documents" not in st.session_state:
#                         index, documents = load_or_create_faiss(DATA_DIR)
#                         st.session_state.documents = documents
#                         init_hybrid(documents, index)
#                     else:
#                         if st.session_state.custom_docs:
#                             init_hybrid(st.session_state.documents, index=None)
#                         else:
#                             index, _ = load_or_create_faiss(DATA_DIR)
#                             init_hybrid(st.session_state.documents, index)

#                     st.write("📚 Retrieving relevant documents...")
#                     contexts, citations, _ = retrieve(query)
#                     progress.progress(30)

#                     st.write("🧠 Generating answer using LLM...")
#                     answer = generate_answer(query, contexts, citations)
#                     progress.progress(70)

#                     st.write("📊 Evaluating answer quality...")
#                     judge = llm_judge(query, answer)
#                     progress.progress(100)

#                     latency = round(time.time() - start, 2)

#                     status.update(label="✅ Completed", state="complete")

#                     # Save result
#                     st.session_state.chat_history.append({
#                         "query": query,
#                         "answer": answer,
#                         "latency": latency,
#                         "judge": judge
#                     })

#                 except Exception as e:
#                     st.error(f"Error: {str(e)}")

#         # 🔥 Unlock UI AFTER processing
#         st.session_state.is_processing = False
#         st.session_state.current_query = ""

#         # 🔥 Rerun to refresh UI cleanly
#         st.rerun()

#     # -----------------------------
#     # DISPLAY CHAT
#     # -----------------------------
#     for chat in reversed(st.session_state.chat_history):
#         st.markdown(f"### 🧑 {chat['query']}")
#         st.markdown(f"**🤖 Answer:** {chat['answer']}")
#         st.caption(f"⏱ {chat['latency']} sec")

#         judge = chat.get("judge", {})

#         if judge:
#             st.markdown("#### 🧠 Evaluation")

#             col1, col2, col3, col4 = st.columns(4)

#             col1.metric("Faithfulness", round(judge.get("faithfulness", 0), 2))
#             col2.metric("Relevance", round(judge.get("relevance", 0), 2))
#             col3.metric("Correctness", round(judge.get("correctness", 0), 2))
#             col4.metric("Score", round(judge.get("final_score", 0), 2))

#             verdict = judge.get("verdict", "")

#             if verdict == "good":
#                 st.success("✅ Good Answer")
#             elif verdict == "partial":
#                 st.warning("⚠️ Partial Answer")
#             elif verdict == "poor":
#                 st.error("❌ Poor Answer")
#             elif verdict == "error":
#                 st.warning("⚠️ Evaluation failed")
                        
#         if judge.get("verdict") == "error":
#             st.warning("⚠️ Evaluation failed (invalid LLM output)")

#         st.markdown("---")

# -----------------------------
# ADMIN PANEL
# -----------------------------
def admin_panel():
    if st.session_state.role != "admin":
        return

    st.subheader("⚙️ Admin Controls")

    if st.button("Reload Base RAG"):
        init_rag_once.clear()
        init_rag_once()
        st.success("Reloaded base dataset!")


# -----------------------------
# MAIN APP
# -----------------------------
# def main_app():
#     sidebar()

#     st.title("🧬 Life Sciences RAG Assistant")

#     # ✅ Initialize only once
#     if not st.session_state.custom_docs:
#         if "documents" not in st.session_state:
#             index, documents = load_or_create_faiss(DATA_DIR)
#             st.session_state.documents = documents
#             init_hybrid(documents, index)

#     tab1, tab2 = st.tabs(["💬 Chat", "📂 Upload"])

#     with tab1:
#         chat_section()

#     with tab2:
#         upload_section()

#     admin_panel()


# # -----------------------------
# # ROUTING
# # -----------------------------
# def main():
#     if st.session_state.token:
#         user, role = verify_token(st.session_state.token)

#         if user:
#             main_app()
#         else:
#             st.warning("Session expired")
#             login_page()
#     else:
#         login_page()


# if __name__ == "__main__":
#     main()


def app_main():
    # 🔥 Sidebar only here (after login)
    # sidebar()
     # 🔥 HEADER WITH LOGOUT
    col1, col2 = st.columns([8, 1])

    with col1:
        st.title("🧬 Life Sciences RAG Assistant")

    with col2:
        if st.button("🚪 Logout"):
            st.session_state.token = None
            st.session_state.role = None
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.custom_docs = False
            st.rerun()

    # st.title("🧬 Life Sciences RAG Assistant")

    # if not st.session_state.custom_docs:
    #     if "documents" not in st.session_state:
    #         index, documents = load_or_create_faiss(DATA_DIR)
    #         st.session_state.documents = documents
    #         init_hybrid(documents, index)

    tab1, tab2 = st.tabs(["💬 Chat", "📂 Upload"])

    with tab1:
        chat_section()

    with tab2:
        upload_section()

    admin_panel()

def main():
    if st.session_state.get("token"):
        user, role = verify_token(st.session_state.token)

        if user:
            # 🔥 SHOW FULL PAGE LOADER FIRST
            if not st.session_state.app_ready:
                with st.spinner("🚀 Loading your workspace..."):
                    time.sleep(0.5)  # small UX delay

                    # Initialize RAG here (blocking)
                    index, documents = load_or_create_faiss(DATA_DIR)
                    st.session_state.documents = documents
                    init_hybrid(documents, index)

                    st.session_state.app_ready = True
                    st.rerun()

            # ✅ Render ONLY after everything ready
            app_main()
        else:
            st.session_state.token = None
            st.warning("Session expired")
            login_page()
    else:
        login_page()

if __name__ == "__main__":
    main()