import os
import gradio as gr
import numpy as np
import datetime

from auth import authenticate, verify_token, check_permission
from embeddings import load_or_create_faiss
from rag import retrieve, generate_answer, init_hybrid
# from audit import log


# =========================
# ✅ PATH SETUP (FIXED)
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

print("DATA_DIR:", DATA_DIR)
print("Exists:", os.path.exists(DATA_DIR))


# =========================
# ✅ LOAD FAISS (ONLY ONCE)
# =========================
try:
    index, documents = load_or_create_faiss(DATA_DIR)
    # Initialize hybrid retrieval
    init_hybrid(documents, index)
    print("FAISS and hybrid search initialized successfully")
except Exception as e:
    print(f"Error loading FAISS: {e}")
    index, documents = None, []
    # Initialize with empty
    index, documents = load_or_create_faiss(DATA_DIR)
    init_hybrid(documents, index)


# =========================
# 🔄 REFRESH EMBEDDINGS (ADMIN ONLY)
# =========================
def refresh_embeddings(token):
    user, role = verify_token(token)
    if not user or role != "admin":
        return "Access denied", "Only admins can refresh embeddings"

    try:
        # Rebuild embeddings
        global index, documents
        index, documents = load_or_create_faiss(DATA_DIR)
        init_hybrid(documents, index)
        return f"Embeddings refreshed at {datetime.datetime.now()}", "Success"
    except Exception as e:
        return f"Error: {str(e)}", "Failed"


# =========================
# 🔐 LOGIN FUNCTION
# =========================
def login(user, pwd):
    try:
        ok, role, token = authenticate(user, pwd)
        if ok:
            admin_visible = (role == "admin")
            print(f"Login successful for {user} with role {role}")
            return (
                "Login successful",
                gr.update(visible=False),
                gr.update(visible=True),
                token,
                role,
                gr.update(visible=admin_visible),
                role,
            )

        print(f"Login failed for {user}")
        return (
            "Invalid login",
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            None,
            gr.update(visible=False),
            "",
        )
    except Exception as e:
        print(f"Exception in login: {e}")
        return (
            f"Login error: {str(e)}",
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            None,
            gr.update(visible=False),
            "",
        )


# =========================
# 💬 CHAT FUNCTION (IMPROVED)
# =========================
def chat(query, token):
    # Verify token
    user, role = verify_token(token)
    if not user:
        return "", "Session expired. Please login again.", 0.0

    # Handle empty query
    if not query:
        return "", "Please enter a question", 0.0

    # Retrieve context
    contexts, citations, scores = retrieve(query)

    # Handle no results
    if not contexts:
        return "", "No relevant documents found", 0.0

    # Generate answer
    answer = generate_answer(query, contexts, citations)

    # Confidence score
    confidence = round(float(0.7 * np.max(scores) + 0.3 * np.mean(scores)), 4)

    # Log interaction (with user)
    # log(user, query, answer)

    # Return results
    return "\n\n---\n\n".join(contexts), answer, confidence


# =========================
# 🔒 LOGOUT FUNCTION
# =========================
def logout(token):
    print(f"Logout requested for token: {token}")
    return (
        "Logged out successfully",
        gr.update(visible=True),
        gr.update(visible=False),
        None,
        None,
        gr.update(visible=False),
        "",
        "",
        0.0,
        "",
    )


# =========================
# 🎨 GRADIO UI
# =========================
with gr.Blocks() as app:
    gr.Markdown("# 🧬 Life Sciences Assistant (Groq + FAISS) Raghava")

    # Login page
    with gr.Column(visible=True) as login_page:
        user = gr.Textbox(label="Username")
        pwd = gr.Textbox(label="Password", type="password")
        btn = gr.Button("Login")
        status = gr.Textbox(label="Status")

    # Chat page (shown after login)
    with gr.Column(visible=False) as chat_page:
        with gr.Row():
            logout_btn = gr.Button("Logout")
            role_display = gr.Textbox(label="Role", interactive=False)
        query = gr.Textbox(label="Ask Question")
        ask = gr.Button("Submit")

        answer = gr.Textbox(label="Answer", lines=10)
        context = gr.Textbox(label="Retrieved Docs", lines=10)
        confidence = gr.Number(label="Confidence")

        with gr.Column(visible=False) as admin_ui:
            gr.Markdown("## 🔧 Admin Panel")
            admin_info = gr.Textbox(label="System Info", lines=5, interactive=False)
            refresh_btn = gr.Button("Refresh Embeddings")
            refresh_status = gr.Textbox(label="Status")

    token_state = gr.State()
    role_state = gr.State()

    btn.click(
        login,
        [user, pwd],
        [status, login_page, chat_page, token_state, role_state, admin_ui, role_display],
    )
    ask.click(chat, [query, token_state], [context, answer, confidence])
    logout_btn.click(
        logout,
        [token_state],
        [status, login_page, chat_page, token_state, role_state, admin_ui, answer, context, confidence, role_display],
    )
    refresh_btn.click(refresh_embeddings, [token_state], [admin_info, refresh_status])


# =========================
# 🚀 LAUNCH APP
# =========================
if __name__ == "__main__":
    app.launch(debug=True)