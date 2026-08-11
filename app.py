import os
import streamlit as st
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Campus Desk | College Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN SYSTEM — "Campus Front Desk" theme
# Ink navy + parchment + brass gold, with a registrar's
# routing-stamp motif for the classifier's decisions.
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
    --bg:#0E141B;
    --bg-elevated:#182029;
    --bg-sidebar:#0A0F14;
    --line:rgba(232,224,208,0.14);
    --parchment:#E8E0D0;
    --parchment-dim:#9AA3B0;
    --brass:#E0B94A;
    --slate:#9AA3B0;
    --academic:#7FBE6E;
    --fee:#E08066;
    --general:#7FA8CC;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* App background */
[data-testid="stAppViewContainer"]{
    background: var(--bg);
}
[data-testid="stHeader"]{
    background: transparent;
}
.stApp, .stApp p, .stApp li, .stApp span, .stApp label { color: var(--parchment); }

/* Sidebar — the "ID card / registrar's counter" */
[data-testid="stSidebar"]{
    background: var(--bg-sidebar);
    color: var(--parchment);
    border-right: 3px solid var(--brass);
}
[data-testid="stSidebar"] * { color: var(--parchment) !important; }
[data-testid="stSidebar"] hr { border-color: var(--line); }

.sidebar-card{
    background: var(--bg-elevated);
    border: 1px solid rgba(224,185,74,0.35);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 14px;
}
.sidebar-eyebrow{
    font-family:'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--brass) !important;
    margin-bottom: 6px;
}
.sidebar-title{
    font-family:'Fraunces', serif;
    font-size: 22px;
    font-weight: 600;
    line-height: 1.2;
    margin-bottom: 2px;
}
.sidebar-sub{
    font-size: 13px;
    color: rgba(232,224,208,0.65) !important;
}

/* Radio group styled like desk-selection tabs */
[data-testid="stSidebar"] [role="radiogroup"] label{
    background: rgba(232,224,208,0.05);
    border: 1px solid rgba(224,185,74,0.3);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    transition: all 0.15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{
    border-color: var(--brass);
    background: rgba(224,185,74,0.1);
}

/* Sidebar buttons */
[data-testid="stSidebar"] button{
    background: transparent !important;
    border: 1px solid var(--brass) !important;
    color: var(--brass) !important;
    border-radius: 8px !important;
    font-family:'IBM Plex Mono', monospace;
    font-size: 12px !important;
    letter-spacing: 0.5px;
}
[data-testid="stSidebar"] button:hover{
    background: var(--brass) !important;
    color: var(--bg-sidebar) !important;
}

/* Hero header */
.hero{
    padding: 6px 4px 18px 4px;
    border-bottom: 2px solid var(--line);
    margin-bottom: 18px;
}
.hero-eyebrow{
    font-family:'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--brass);
    margin-bottom: 6px;
}
.hero-title{
    font-family:'Fraunces', serif;
    font-size: 40px;
    font-weight: 600;
    color: var(--parchment);
    margin: 0;
    line-height: 1.1;
}
.hero-sub{
    font-size: 15px;
    color: var(--slate);
    margin-top: 8px;
    max-width: 640px;
}

/* Three "desks" strip — encodes the actual routing paths */
.desk-strip{
    display:flex;
    gap: 10px;
    margin: 16px 0 22px 0;
    flex-wrap: wrap;
}
.desk-chip{
    font-family:'IBM Plex Mono', monospace;
    font-size: 12px;
    padding: 7px 14px;
    border-radius: 999px;
    border: 1.5px dashed;
    letter-spacing: 0.3px;
}
.desk-academic{ color: var(--academic); border-color: var(--academic); background: rgba(127,190,110,0.08); }
.desk-fee{ color: var(--fee); border-color: var(--fee); background: rgba(224,128,102,0.08); }
.desk-general{ color: var(--general); border-color: var(--general); background: rgba(127,168,204,0.08); }

/* Chat bubbles */
[data-testid="stChatMessage"]{
    background: var(--bg-elevated);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 4px 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
}
[data-testid="stChatMessage"] p{ color: var(--parchment); }

/* Routing stamp shown under assistant replies */
.stamp{
    display:inline-block;
    font-family:'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
    border: 1.5px dashed;
    transform: rotate(-1.5deg);
    margin-top: 6px;
}
.stamp-academic{ color: var(--academic); border-color: var(--academic); }
.stamp-fee{ color: var(--fee); border-color: var(--fee); }
.stamp-general{ color: var(--general); border-color: var(--general); }

/* Chat input */
[data-testid="stChatInput"]{
    background: var(--bg-elevated) !important;
    border: 2px solid var(--brass) !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea{
    color: var(--parchment) !important;
    background: transparent !important;
}

/* Info banner */
[data-testid="stAlert"]{
    background: var(--bg-elevated) !important;
    color: var(--parchment) !important;
    border: 1px solid var(--line) !important;
}

#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# ENV CHECK
# ============================================================
if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "⚠️ **GROQ_API_KEY** was not found. Add it to a `.env` file "
        "in this project's folder, e.g.:\n\n`GROQ_API_KEY=your_key_here`"
    )
    st.stop()

# ============================================================
# RAG RETRIEVERS
# (cached with st.cache_resource so the PDFs are only loaded /
#  embedded / indexed once per app session, not on every rerun —
#  the retrieval + graph logic itself is untouched)
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


embeddings = get_embeddings()


@st.cache_resource(show_spinner="Indexing college documents...")
def build_retriver(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 4})


acedmic_retriver = build_retriver("academics_handbook.pdf")
fees_retriver = build_retriver("fee_structure.pdf")


llm = ChatGroq(model="llama-3.3-70b-versatile",
               temperature=0.4)


# ============================================================
# STATE
# ============================================================
class state(TypedDict):
    program: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrived_context: str


# ============================================================
# NODES  (unchanged from the original script)
# ============================================================

def classifier_node(state: state) -> dict:
    """Look at the latest message and decide which path to take."""

    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )
    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def academic_rag_node(state: state) -> dict:
    """Retrives relevant cunks from the academic handbook"""
    query = state["messages"][-1].content
    docs = acedmic_retriver.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrived_context": context}


def fee_rag_node(state: state) -> dict:
    """Retrives relevent chunks from fee structure PDF."""

    query = state["messages"][-1].content
    docs = fees_retriver.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrived_context": context}


def general_node(state: state) -> dict:
    """Answers directly using the LLM's own knoladge , no retrival needed."""
    return {"retrived_context": "NO_RETRIVAL_NEEDED"}


def response_node(state: state) -> dict:
    """Generates the final answer , personalized using the students program """
    query = state["messages"][-1].content
    programme = state.get("program", "unknown")
    context = state["retrived_context"]
    if context == "NO_RETRIVAL_NEEDED":
        prompt = (
            f"you are a friendly colleg assistant talking to a {programme} student."
            f"Answer this question using own general knoledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


def router_query(state: state):
    if state["query_type"] == 'academic':
        return "academic_rag"
    elif state["query_type"] == "fee":
        return "fee_rag"
    else:
        return "general"


# ============================================================
# GRAPH  (unchanged, wrapped in a cache_resource so it's only
# compiled once)
# ============================================================
@st.cache_resource
def get_app():
    graph = StateGraph(state)
    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges(
        "classifier", router_query
    )

    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")

    graph.add_edge("response", END)

    return graph.compile()


app = get_app()

# ============================================================
# UI STATE
# ============================================================
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []
if "student_program" not in st.session_state:
    st.session_state.student_program = "BCA"

STAMP_META = {
    "academic": ("stamp-academic", "🎓 routed → academic desk"),
    "fee": ("stamp-fee", "💰 routed → fee desk"),
    "general": ("stamp-general", "💬 routed → general desk"),
}

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-eyebrow">Student Record</div>
            <div class="sidebar-title">Campus Desk</div>
            <div class="sidebar-sub">Multi-agent college assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-eyebrow" style="margin-top:4px;">Your Programme</div>', unsafe_allow_html=True)
    program_options = ["BCA", "BBA", "B COM (H)"]
    st.session_state.student_program = st.radio(
        label="Programme",
        options=program_options,
        index=program_options.index(st.session_state.student_program),
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-eyebrow">How routing works</div>
        <div class="sidebar-sub" style="line-height:1.7;">
        🎓 <b>Academic desk</b> — attendance, exams, credits, promotion<br/>
        💰 <b>Fee desk</b> — tuition, refunds, scholarships<br/>
        💬 <b>General desk</b> — everything else
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.chat_display = []
        st.rerun()

# ============================================================
# HERO
# ============================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-eyebrow">Front Desk · {st.session_state.student_program} Student</div>
        <div class="hero-title">College Assistant</div>
        <div class="hero-sub">Ask about academics, fees, or anything else — your question
        gets routed to the right desk automatically.</div>
    </div>
    <div class="desk-strip">
        <span class="desk-chip desk-academic">🎓 Academic Desk</span>
        <span class="desk-chip desk-fee">💰 Fee Desk</span>
        <span class="desk-chip desk-general">💬 General Desk</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CHAT HISTORY
# ============================================================
for msg in st.session_state.chat_display:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🏛️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("category"):
            css_class, label = STAMP_META.get(msg["category"], ("stamp-general", msg["category"]))
            st.markdown(f'<span class="stamp {css_class}">{label}</span>', unsafe_allow_html=True)

if not st.session_state.chat_display:
    st.info("👋 No messages yet — ask something like *\"What's the attendance policy?\"* or *\"When is the next fee installment due?\"*")

# ============================================================
# CHAT INPUT  →  invokes the LangGraph app exactly as the
# original CLI loop did (one fresh human message per turn)
# ============================================================
user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state.chat_display.append({"role": "user", "content": user_query})

    with st.spinner("🤔 Thinking..."):
        result = app.invoke({
            "program": st.session_state.student_program,
            "messages": [("human", user_query)]
        })

    answer = result["messages"][-1].content
    category = result.get("query_type", "general")

    st.session_state.chat_display.append({
        "role": "assistant",
        "content": answer,
        "category": category,
    })

    st.rerun()