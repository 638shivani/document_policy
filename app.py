import json
import os
import re
import numpy as np
import PyPDF2
import streamlit as st
import google.generativeai as genai

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_DEPS_OK = True
except ImportError as e:
    RAG_DEPS_OK = False
    RAG_IMPORT_ERR = str(e)

# ---------------- CONFIG & OBSIDIAN DARK CSS ----------------
st.set_page_config(page_title="PolicyMind Pro", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    /* Global Obsidian Background */
    .stApp, [data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
        background-color: #080A0D !important;
        color: #E5E7EB !important;
    }

    /* LANGUAGE SELECTBOX FIX (Matches Dark UI) */
    div[data-baseweb="select"] > div {
        background-color: #0F1217 !important;
        color: white !important;
        border: 1px solid #30363D !important;
        border-radius: 8px;
    }
    div[role="listbox"] {
        background-color: #0F1217 !important;
        border: 1px solid #30363D !important;
    }
    div[role="option"] {
        background-color: #0F1217 !important;
        color: white !important;
    }
    div[role="option"]:hover {
        background-color: #1F2937 !important;
    }

    /* Dashboard & Metric Cards */
    .metric-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Neon Accents */
    h1, h2, h3, h4 { color: #60A5FA !important; }
    .stButton>button {
        background: linear-gradient(90deg, #1E40AF, #3B82F6);
        color: white !important; border: none !important; border-radius: 8px;
        width: 100%; transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0px 0px 15px rgba(59, 130, 246, 0.6);
        transform: translateY(-2px);
    }

    /* Text Inputs & Tabs */
    input {
        background-color: #0F1217 !important;
        color: white !important;
        border: 1px solid #30363D !important;
    }
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; }
    .stTabs [aria-selected="true"] { color: #60A5FA !important; border-bottom-color: #60A5FA !important; }
    </style>
    """, unsafe_allow_html=True)

# ---------------- MODEL & EMBEDDING LOADERS ----------------
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ GEMINI_API_KEY Missing in Secrets / Environment Variables")
    st.stop()

genai.configure(api_key=API_KEY)

@st.cache_resource
def load_stable_model():
    """Tries preferred Gemini models and falls back to available generateContent models."""
    preferred = ["gemini-3.1-flash-lite", "gemini-3-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    try:
        allowed = [
            m.name for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        for p in preferred:
            for a in allowed:
                if p in a:
                    m = genai.GenerativeModel(a)
                    m.generate_content("ping", generation_config={"max_output_tokens": 1})
                    return m, a
        if allowed:
            m = genai.GenerativeModel(allowed[0])
            return m, allowed[0]
    except Exception as e:
        return None, str(e)
    return None, "No models available"

@st.cache_resource
def load_embed_model():
    """Caches SentenceTransformer so embeddings are not reloaded on every rerun."""
    if not RAG_DEPS_OK:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")

model, model_id = load_stable_model()
embed_model = load_embed_model()

# ---------------- MULTILINGUAL DATA ----------------
LANG_DB = {
    "English": {
        "title": "🧠 PolicyMind Pro v5.0 (RAG)",
        "intro": "RAG-powered insurance policy analysis with vector retrieval.",
        "how": "How to use",
        "steps": ["Upload Policy PDF (Indexed with FAISS)", "Ask a Medical Scenario", "Analyze Retrieved Coverage"],
        "examples_title": "💡 Example Scenarios",
        "examples": [
            "I have had this policy for 2 years, is dental checkup covered?",
            "Is emergency appendectomy covered for a 30-year-old?"
        ],
        "stats": ["Total Requests", "Approved", "Rejected"],
        "labels": ["Decision", "Certainty", "Policy Age", "Procedure"]
    },
    "Hindi": {
        "title": "🧠 पॉलिसीमाइंड प्रो v5.0 (RAG)",
        "intro": "वेक्टर रिट्रीवल (RAG) के साथ बीमा पॉलिसी का विश्लेषण।",
        "how": "उपयोग कैसे करें",
        "steps": ["पॉलिसी PDF अपलोड करें", "एक चिकित्सा स्थिति पूछें", "कवरेज का विश्लेषण करें"],
        "examples_title": "💡 उदाहरण प्रश्न",
        "examples": [
            "मेरे पास यह पॉलिसी 2 साल से है, क्या डेंटल चेकअप कवर है?",
            "क्या 30 वर्षीय व्यक्ति के लिए आपातकालीन अपेंडक्टोमी कवर है?"
        ],
        "stats": ["कुल अनुरोध", "स्वीकृत", "अस्वीकृत"],
        "labels": ["निर्णय", "विश्वास", "पॉलिसी की आयु", "प्रक्रिया"]
    },
    "Kannada": {
        "title": "🧠 ಪಾಲಿಸಿಮೈಂಡ್ ಪ್ರೊ v5.0 (RAG)",
        "intro": "RAG ವೆಕ್ಟರ್ ಹುಡುಕಾಟದೊಂದಿಗೆ ವಿಮಾ ಪಾಲಿಸಿ ವಿಶ್ಲೇಷಣೆ.",
        "how": "ಬಳಸುವುದು ಹೇಗೆ",
        "steps": ["ಪಾಲಿಸಿ PDF ಅಪ್‌ಲೋಡ್ ಮಾಡಿ", "ವೈದ್ಯಕೀಯ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ", "ಕವರೇಜ್ ವಿಶ್ಲೇಷಿಸಿ"],
        "examples_title": "💡 ಉದಾಹರಣೆ ಪ್ರಶ್ನೆಗಳು",
        "examples": [
            "ನನ್ನ ಬಳಿ ಈ ಪಾಲಿಸಿ 2 ವರ್ಷಗಳಿಂದ ಇದೆ, ಹಲ್ಲಿನ ತಪಾಸಣೆ ಕವರ್ ಆಗುತ್ತದೆಯೇ?",
            "30 ವರ್ಷದ ವ್ಯಕ್ತಿಗೆ ತುರ್ತು ಅಪೆಂಡೆಕ್ಟೊಮಿ ಕವರ್ ಆಗುತ್ತದೆಯೇ?"
        ],
        "stats": ["ಒಟ್ಟು ವಿನಂತಿಗಳು", "ಅನುಮೋದಿಸಲಾಗಿದೆ", "ತಿರಸ್ಕರಿಸಲಾಗಿದೆ"],
        "labels": ["ತೀರ್ಮಾನ", "ನಂಬಿಕೆ", "ಪಾಲಿಸಿ ಅವಧಿ", "ಕಾರ್ಯವಿಧಾನ"]
    }
}

# ---------------- SIDEBAR ----------------
st.sidebar.title("Settings")
sel_lang = st.sidebar.selectbox("🌐 Select Language", ["English", "Hindi", "Kannada"])
top_k = st.sidebar.slider("🔎 Retrieved Chunks (top_k)", min_value=1, max_value=8, value=3)
L = LANG_DB[sel_lang]

if model:
    st.sidebar.success(f"⚡ Connected to: {model_id}")
else:
    st.sidebar.error(f"🚨 AI Handshake Failed: {model_id}")

if RAG_DEPS_OK and embed_model:
    st.sidebar.success("🧩 RAG Engine: FAISS + MiniLM Ready")
else:
    st.sidebar.error(f"🚨 RAG Missing Deps: {RAG_IMPORT_ERR if not RAG_DEPS_OK else 'Init failed'}")

# ---------------- STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "latest" not in st.session_state:
    st.session_state.latest = None

# ---------------- RAG LOGIC ----------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(cleaned)
    # Fallback chunker if langchain splitter isn't installed
    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(cleaned), step):
        chunks.append(cleaned[start:start + chunk_size])
    return chunks

def create_vector_store(chunks):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_chunks(query, chunks, index, k=3):
    if not chunks or index is None:
        return []
    k = min(k, len(chunks))
    query_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]

def analyze_claim_rag(query, chunks, index, lang_name, k=3):
    relevant_chunks = retrieve_chunks(query, chunks, index, k=k)
    context = "\n---\n".join(relevant_chunks)

    prompt = f"""
    You are an insurance policy analyst. Base your decision and explanation ONLY on the retrieved Policy Context below.
    If the policy context does not contain enough information, state that clearly in "reason".

    Retrieved Policy Context:
    {context}

    User Scenario:
    {query}

    Respond in {lang_name}. Output STRICT JSON ONLY (no markdown fences):
    {{
        "decision": "Approved/Rejected",
        "reason": "Clear explanation grounded only in the retrieved policy context",
        "conf": "X%",
        "age": "Detected policy/user age or N/A",
        "proc": "Medical procedure or N/A"
    }}
    """
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
        data = json.loads(raw)
        data["retrieved_chunks"] = relevant_chunks
        return data
    except Exception as e:
        return {
            "decision": "Error",
            "reason": f"Parsing/AI error: {e}",
            "conf": "0%",
            "age": "N/A",
            "proc": "N/A",
            "retrieved_chunks": relevant_chunks
        }

# ---------------- UI: HOME DASHBOARD ----------------
st.title(L["title"])
st.caption(L["intro"])

with st.container():
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(f"🔍 {L['how']}")
        for step in L["steps"]:
            st.write(f"🔹 {step}")
    with c2:
        st.markdown(f"#### {L['examples_title']}")
        for ex in L["examples"]:
            st.info(ex)

st.divider()

if not model or not embed_model:
    st.stop()

# ---------------- MAIN TABS ----------------
t1, t2, t3 = st.tabs(["📄 Doc Center", "🧠 AI Analysis", "📊 Analytics & History"])

with t1:
    up = st.file_uploader("Upload Policy PDF", type="pdf")
    if up is not None and up.name != st.session_state.pdf_name:
        with st.spinner("Extracting text, chunking, and building FAISS index..."):
            text = extract_text(up)
            chunks = chunk_text(text)
            if not chunks:
                st.error("Could not extract readable text from this PDF.")
            else:
                st.session_state.pdf_name = up.name
                st.session_state.chunks = chunks
                st.session_state.index = create_vector_store(chunks)
                st.success(f"Policy Indexed: {len(chunks)} chunks ready in FAISS.")
    elif st.session_state.chunks:
        st.info(f"Active Policy: **{st.session_state.pdf_name}** ({len(st.session_state.chunks)} chunks)")

    q = st.text_input("Enter your coverage query:")
    if st.button("🚀 Analyze Now"):
        if st.session_state.chunks and st.session_state.index is not None and q.strip():
            with st.spinner("Retrieving clauses & analyzing coverage..."):
                res = analyze_claim_rag(
                    q.strip(),
                    st.session_state.chunks,
                    st.session_state.index,
                    sel_lang,
                    k=top_k
                )
                st.session_state.latest = res
                st.session_state.history.append(res)
                st.toast("Analysis complete!")
        else:
            st.warning("Upload a readable PDF and enter a query first.")

with t2:
    if st.session_state.latest:
        r = st.session_state.latest
        approved_vals = {"Approved", "ಅನುಮೋದಿಸಲಾಗಿದೆ", "स्वीकृत"}
        color = "#10B981" if r.get("decision") in approved_vals else "#EF4444"

        st.markdown(f"""
        <div style="border-left: 10px solid {color}; background-color: #111827; padding: 25px; border-radius: 12px; margin-bottom: 25px;">
            <h1 style="color: {color} !important; margin: 0;">{r.get('decision', 'N/A')}</h1>
            <p style="color: #D1D5DB; font-size: 1.1em; margin-top: 10px;">{r.get('reason', '')}</p>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(4)
        m_vals = [r.get("decision", "N/A"), r.get("conf", "0%"), r.get("age", "N/A"), r.get("proc", "N/A")]
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #9CA3AF; font-size: 0.8em; text-transform: uppercase;">{L['labels'][i]}</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: #60A5FA;">{m_vals[i]}</div>
                </div>
                """, unsafe_allow_html=True)

        if r.get("retrieved_chunks"):
            st.divider()
            st.subheader("📚 Retrieved Policy Excerpts")
            for idx, ch in enumerate(r["retrieved_chunks"], 1):
                with st.expander(f"Chunk #{idx}"):
                    st.write(ch)
    else:
        st.info("No active result yet. Run a query in Doc Center.")

with t3:
    approved_vals = {"Approved", "ಅನುಮೋದಿಸಲಾಗಿದೆ", "स्वीकृत"}
    total = len(st.session_state.history)
    appr = sum(1 for x in st.session_state.history if x.get("decision") in approved_vals)
    rej = total - appr

    st.subheader(f"📊 {L['title']} Stats")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(f'<div class="metric-card"><div style="color:#9CA3AF;">{L["stats"][0]}</div><div style="font-size:2em; font-weight:bold;">{total}</div></div>', unsafe_allow_html=True)
    with sc2:
        st.markdown(f'<div class="metric-card"><div style="color:#9CA3AF;">{L["stats"][1]}</div><div style="font-size:2em; font-weight:bold; color:#10B981;">{appr}</div></div>', unsafe_allow_html=True)
    with sc3:
        st.markdown(f'<div class="metric-card"><div style="color:#9CA3AF;">{L["stats"][2]}</div><div style="font-size:2em; font-weight:bold; color:#EF4444;">{rej}</div></div>', unsafe_allow_html=True)

    st.divider()
    for item in reversed(st.session_state.history):
        with st.expander(f"🕒 {item.get('proc', 'Query')} - {item.get('decision', 'N/A')}"):
            st.write(f"**Justification:** {item.get('reason', '')}")
            st.write(f"**Confidence:** {item.get('conf', '0%')}")
