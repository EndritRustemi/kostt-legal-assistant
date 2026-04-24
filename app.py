"""
KOSTT Legal Assistant — RAG System
"""

import streamlit as st
from pathlib import Path

from rag.ingest import build_index, count_pdfs, CATEGORIES
from rag.retriever import retrieve
from rag.llm import generate_answer

LAWS_DIR = Path(__file__).parent / "data" / "laws"

st.set_page_config(
    page_title="KOSTT Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .source-card {
        background: #f4f7fb;
        border-left: 4px solid #1a5276;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .score-badge {
        background: #d5f5e3;
        color: #1e8449;
        padding: 1px 7px;
        border-radius: 10px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    /* Fshih header-in e Streamlit */
    #MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return st.session_state.get("api_key", "")


@st.cache_resource(show_spinner="Duke indeksuar dokumentet (vetëm herën e parë)...")
def load_index():
    return build_index(LAWS_DIR)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚖️ KOSTT Legal")
    st.markdown("---")

    # API Key
    api_key = get_api_key()
    if not api_key:
        entered = st.text_input("🔑 Gemini API Key", type="password")
        if entered:
            st.session_state["api_key"] = entered
            st.rerun()
    else:
        st.success("✅ API Key i konfiguruar")

    st.markdown("---")

    # Statistikat
    st.markdown("### 📂 Dokumentet")
    counts = count_pdfs(LAWS_DIR)
    total = sum(counts.values())
    for cat, n in counts.items():
        st.markdown(f"{'✅' if n > 0 else '⬜'} **{cat}**: {n} dok.")
    st.markdown(f"**Gjithsej: {total}**")

    st.markdown("---")

    # Ngarko dokument
    with st.expander("📥 Ngarko dokument"):
        uploaded = st.file_uploader("PDF", type="pdf", accept_multiple_files=True,
                                    label_visibility="collapsed")
        cat_labels = list(CATEGORIES.values())
        cat_keys   = list(CATEGORIES.keys())
        sel_label  = st.selectbox("Kategoria", cat_labels, label_visibility="collapsed")
        sel_key    = cat_keys[cat_labels.index(sel_label)]

        if uploaded and st.button("Ngarko", type="primary", use_container_width=True):
            dest = LAWS_DIR / sel_key
            dest.mkdir(parents=True, exist_ok=True)
            for f in uploaded:
                (dest / f.name).write_bytes(f.read())
            st.cache_resource.clear()
            st.success(f"✅ {len(uploaded)} dok. u ngarkuan.")
            st.rerun()

    # Shfaq dokumentet
    with st.expander("📋 Lista e dokumenteve"):
        for cat_key, cat_label in CATEGORIES.items():
            folder = LAWS_DIR / cat_key
            if folder.exists():
                pdfs = sorted(folder.glob("*.pdf"))
                for pdf in pdfs:
                    c1, c2 = st.columns([5, 1])
                    c1.caption(pdf.name)
                    if c2.button("🗑️", key=f"del_{pdf.name}"):
                        pdf.unlink()
                        st.cache_resource.clear()
                        st.rerun()

    st.markdown("---")
    col1, col2 = st.columns(2)
    if col1.button("🔄 Ri-indekso", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    if col2.button("🗑️ Pastro", use_container_width=True, help="Fshi historikun e bisedës"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("KOSTT Sh.A. · Sektori Juridik")


# ── CHAT ──────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mesazhi mirëpritës kur biseda është bosh
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center; padding: 80px 0 40px 0; color: #888;'>
        <div style='font-size:2.5rem'>⚖️</div>
        <div style='font-size:1.3rem; font-weight:600; color:#1a3a5c; margin:10px 0'>KOSTT Legal Assistant</div>
        <div style='font-size:0.9rem'>Pyetje juridike bazuar në ligjet e energjisë<br>Kosovë · ZRRE · ENTSO-E · EU</div>
    </div>
    """, unsafe_allow_html=True)

# Shfaq historikun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Burimet ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 <b>{src["doc"]}</b> · {src["category"]} · Faqja {src["page"]}'
                        f' <span class="score-badge">{src["score"]}</span><br>'
                        f'<i>{src["snippet"]}...</i></div>',
                        unsafe_allow_html=True,
                    )

# Input — Streamlit e mban automatikisht në fund të faqes
if question := st.chat_input("Shkruani pyetjen tuaj juridike..."):
    api_key = get_api_key()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            st.warning("⚠️ Vendosni API Key në sidebar.")
        elif sum(count_pdfs(LAWS_DIR).values()) == 0:
            st.warning("⚠️ Ngarkoni dokumente ligjore nga sidebar.")
        else:
            with st.spinner("Duke kërkuar..."):
                try:
                    index  = load_index()
                    chunks = retrieve(index, question, api_key, top_k=5)
                    answer, sources, src_type = generate_answer(question, chunks, api_key)

                    if src_type == "web":
                        st.info("🌐 Nuk u gjet në dokumentet tuaja — po kërkohet në internet. Verifikoni me burime zyrtare.")
                    elif src_type == "documents":
                        st.success("📄 U gjet në dokumentet tuaja.")

                    st.markdown(answer)

                    if sources:
                        label = "🌐 Burimet nga Interneti" if src_type == "web" else f"📚 Burimet ({len(sources)})"
                        with st.expander(label):
                            for src in sources:
                                if src_type == "web" and src.get("url"):
                                    st.markdown(
                                        f'<div class="source-card">🌐 <b>'
                                        f'<a href="{src["url"]}" target="_blank">{src["doc"]}</a></b><br>'
                                        f'<i>{src["snippet"]}...</i></div>',
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(
                                        f'<div class="source-card">📄 <b>{src["doc"]}</b> · '
                                        f'{src["category"]} · Faqja {src["page"]}'
                                        f' <span class="score-badge">{src["score"]}</span><br>'
                                        f'<i>{src["snippet"]}...</i></div>',
                                        unsafe_allow_html=True,
                                    )

                    st.session_state.messages.append({
                        "role": "assistant", "content": answer,
                        "sources": sources, "src_type": src_type,
                    })
                except Exception as e:
                    st.error(f"Gabim: {e}")
