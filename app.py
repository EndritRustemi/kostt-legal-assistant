"""
Energy Law & Legal AI by ENDRIT — RAG System
"""

import os
import shutil
from datetime import datetime, timedelta, timezone

import streamlit as st
from pathlib import Path

from rag.indexer import build_index, count_pdfs, CATEGORIES
from rag.retriever import retrieve
from rag.llm import generate_answer
from rag.web_scraper import scrape_url, load_web_sources, save_web_sources, KNOWN_SOURCES

LAWS_DIR         = Path(__file__).parent / "data" / "laws"
WEB_SOURCES_PATH = Path(__file__).parent / "data" / "web_sources.json"
REFRESH_CFG_PATH = Path(__file__).parent / "data" / "refresh_cfg.txt"

_REFRESH_OPTIONS = {
    "Çdo 1 orë":   1,
    "Çdo 3 orë":   3,
    "Çdo 6 orë":   6,
    "Çdo 12 orë":  12,
    "Çdo 24 orë":  24,
}


def _load_refresh_hours() -> int:
    try:
        return int(REFRESH_CFG_PATH.read_text().strip())
    except Exception:
        return 6


def _save_refresh_hours(hours: int) -> None:
    REFRESH_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFRESH_CFG_PATH.write_text(str(hours))

st.set_page_config(
    page_title="Energy Law & Legal AI by ENDRIT",
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
    #MainMenu, header, footer { visibility: hidden; }
    /* edit button */
    div[data-testid="stHorizontalBlock"] > div:has(> div[data-testid="stButton"] > button[kind="secondary"]) {
        display: flex;
        align-items: flex-end;
    }
    button.edit-msg-btn {
        background: none !important;
        border: none !important;
        color: #aaa !important;
        font-size: 0.78rem !important;
        padding: 2px 6px !important;
        cursor: pointer;
    }
    button.edit-msg-btn:hover { color: #1a5276 !important; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _secret(key: str) -> str:
    """Read secret: env var first (HF Spaces Docker), then st.secrets (local)."""
    val = os.environ.get(key, "")
    if val:
        return val
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def get_api_key() -> str:
    api = _secret("ANTHROPIC_API_KEY")
    return api or st.session_state.get("api_key", "")


# ── STARTUP: sync PDFs from HF Dataset repo ───────────────────────────────────

@st.cache_resource(show_spinner="Duke ngarkuar dokumentet...")
def _sync_pdfs() -> str:
    hf_token   = _secret("HF_TOKEN")
    hf_dataset = _secret("HF_DATASET_ID")

    if not hf_token:
        return "MISSING:HF_TOKEN"
    if not hf_dataset:
        return "MISSING:HF_DATASET_ID"

    try:
        from huggingface_hub import snapshot_download
        tmp = "/tmp/hf_legal_docs"
        snapshot_download(
            repo_id=hf_dataset,
            repo_type="dataset",
            token=hf_token,
            local_dir=tmp,
            local_dir_use_symlinks=False,
        )
        app_dir = Path(__file__).parent
        count = 0
        for pdf in Path(tmp).rglob("*.pdf"):
            rel  = pdf.relative_to(tmp)
            dest = app_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf, dest)
            count += 1
        return f"OK:{count}"
    except Exception as e:
        return f"ERR:{e}"

_SYNC_STATUS = _sync_pdfs()


@st.cache_resource(show_spinner="Duke ngarkuar indeksin e dokumenteve...")
def load_index():
    """Build (or reload from disk cache) the ChromaDB index."""
    col      = build_index(LAWS_DIR)
    built_at = datetime.now()
    return col, built_at


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚖️ Energy Law & Legal AI")
    st.markdown("---")

    api_key = get_api_key()
    if api_key and api_key.startswith("AIzaSy"):
        # Stored key is a Google/Gemini key — clear it and force re-entry
        st.session_state.pop("api_key", None)
        api_key = ""
        st.error("⚠️ Çelësi i ruajtur ishte Google/Gemini. Ju lutem vendosni çelësin Anthropic (fillon me **sk-ant-**).")
    if not api_key:
        entered = st.text_input("🔑 Anthropic API Key", type="password",
                                placeholder="sk-ant-...")
        if entered:
            if entered.startswith("AIzaSy"):
                st.error("❌ Ky duket si çelës Google/Gemini. Ju lutem vendosni çelësin Anthropic (fillon me **sk-ant-**).")
            else:
                st.session_state["api_key"] = entered
                st.rerun()
    else:
        st.success("✅ API Key Anthropic i konfiguruar")

    st.markdown("---")

    st.markdown("### 📂 Dokumentet")
    if _SYNC_STATUS.startswith("ERR") or _SYNC_STATUS.startswith("MISSING"):
        st.warning(f"⚠️ Sync: {_SYNC_STATUS}")
    counts = count_pdfs(LAWS_DIR)
    total  = sum(counts.values())
    for cat_key, n in counts.items():
        cat_label = CATEGORIES.get(cat_key, cat_key)
        icon = "✅" if n > 0 else "⬜"
        st.markdown(f"{icon} **{cat_label}**: {n} dok.")
    st.markdown(f"**Gjithsej: {total} dokument(e)**")

    st.markdown("---")

    with st.expander("📥 Ngarko dokument"):
        uploaded  = st.file_uploader("PDF", type="pdf", accept_multiple_files=True,
                                     label_visibility="collapsed")
        cat_labels = list(CATEGORIES.values())
        cat_keys   = list(CATEGORIES.keys())
        sel_label  = st.selectbox("Kategoria", cat_labels, label_visibility="collapsed")
        sel_key    = cat_keys[cat_labels.index(sel_label)]

        if uploaded:
            if st.button("Ngarko", type="primary", use_container_width=True):
                dest = LAWS_DIR / sel_key
                dest.mkdir(parents=True, exist_ok=True)
                hf_token   = _secret("HF_TOKEN")
                hf_dataset = _secret("HF_DATASET_ID")
                saved, hf_errors = [], []

                for f in uploaded:
                    try:
                        file_bytes = f.read()
                        (dest / f.name).write_bytes(file_bytes)
                        saved.append((f.name, file_bytes))
                    except Exception as e:
                        hf_errors.append(f"Shkrim '{f.name}': {e}")

                for fname, fbytes in saved:
                    if hf_token and hf_dataset:
                        try:
                            from huggingface_hub import HfApi
                            HfApi().upload_file(
                                path_or_fileobj=fbytes,
                                path_in_repo=f"data/laws/{sel_key}/{fname}",
                                repo_id=hf_dataset,
                                repo_type="dataset",
                                token=hf_token,
                            )
                        except Exception as e:
                            hf_errors.append(f"HF ruajtje '{fname}': {e}")

                if hf_errors:
                    st.session_state["_upload_err"] = "\n".join(hf_errors)
                else:
                    st.session_state["_upload_ok"] = len(saved)
                st.cache_resource.clear()
                st.rerun()

        if "_upload_ok" in st.session_state:
            st.success(f"✅ {st.session_state.pop('_upload_ok')} dok. u ngarkuan.")
        if "_upload_err" in st.session_state:
            st.error(st.session_state.pop("_upload_err"))

    with st.expander("📋 Lista e dokumenteve"):
        def _fmt_size(path):
            b = path.stat().st_size
            return f"{b/1024:.0f} KB" if b < 1024*1024 else f"{b/1024/1024:.1f} MB"

        def _hf_delete(cat_k, fname):
            tok = _secret("HF_TOKEN"); ds = _secret("HF_DATASET_ID")
            if tok and ds:
                try:
                    from huggingface_hub import HfApi
                    HfApi().delete_file(
                        path_in_repo=f"data/laws/{cat_k}/{fname}",
                        repo_id=ds, repo_type="dataset", token=tok,
                    )
                except Exception:
                    pass

        def _hf_upload(cat_k, fname, fbytes):
            tok = _secret("HF_TOKEN"); ds = _secret("HF_DATASET_ID")
            if tok and ds:
                try:
                    from huggingface_hub import HfApi
                    HfApi().upload_file(
                        path_or_fileobj=fbytes,
                        path_in_repo=f"data/laws/{cat_k}/{fname}",
                        repo_id=ds, repo_type="dataset", token=tok,
                    )
                except Exception:
                    pass

        has_any = False
        for cat_key, cat_label in CATEGORIES.items():
            folder = LAWS_DIR / cat_key
            if not folder.exists():
                continue
            pdfs = sorted(folder.glob("*.pdf"))
            if not pdfs:
                continue
            has_any = True
            st.markdown(f"**📁 {cat_label}** — {len(pdfs)} dok.")
            for pdf in pdfs:
                moving_key = f"_moving_{cat_key}_{pdf.name}"
                if st.session_state.get(moving_key):
                    st.markdown(f"↗️ **{pdf.stem[:32]}**")
                    other_cats = {k: v for k, v in CATEGORIES.items() if k != cat_key}
                    new_cat = st.selectbox(
                        "Zhvendos në kategorinë:",
                        list(other_cats.keys()),
                        format_func=lambda k: CATEGORIES[k],
                        key=f"sel_{cat_key}_{pdf.name}",
                        label_visibility="collapsed",
                    )
                    mc1, mc2 = st.columns(2)
                    if mc1.button("✓ Konfirmo", key=f"ok_mv_{cat_key}_{pdf.name}",
                                  type="primary", use_container_width=True):
                        fbytes = pdf.read_bytes()
                        new_dest = LAWS_DIR / new_cat / pdf.name
                        new_dest.parent.mkdir(parents=True, exist_ok=True)
                        new_dest.write_bytes(fbytes)
                        pdf.unlink()
                        _hf_upload(new_cat, pdf.name, fbytes)
                        _hf_delete(cat_key, pdf.name)
                        st.session_state.pop(moving_key, None)
                        st.cache_resource.clear()
                        st.rerun()
                    if mc2.button("✕ Anulo", key=f"no_mv_{cat_key}_{pdf.name}",
                                  use_container_width=True):
                        st.session_state.pop(moving_key, None)
                        st.rerun()
                else:
                    c1, c2, c3 = st.columns([7, 1, 1])
                    name_short = pdf.stem[:28] + ("…" if len(pdf.stem) > 28 else "")
                    c1.markdown(
                        f"<span style='font-size:0.82rem'>📄 {name_short}</span>"
                        f"<br><span style='font-size:0.72rem;color:#888'>{_fmt_size(pdf)}</span>",
                        unsafe_allow_html=True,
                    )
                    if c2.button("↗️", key=f"mv_{cat_key}_{pdf.name}", help="Zhvendos"):
                        st.session_state[moving_key] = True
                        st.rerun()
                    if c3.button("🗑️", key=f"del_{cat_key}_{pdf.name}", help="Fshi"):
                        pdf.unlink()
                        _hf_delete(cat_key, pdf.name)
                        st.cache_resource.clear()
                        st.rerun()
            st.markdown("---")

        if not has_any:
            st.caption("Asnjë dokument i ngarkuar.")

    # ── Web Sources ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌐 Burime Web")

    if st.session_state.get("_index_built_at"):
        _built_at: datetime = st.session_state["_index_built_at"]
        st.caption(f"🕐 Indeksi ndërtuar: {_built_at.strftime('%d %b %Y, %H:%M')}")

    web_sources = load_web_sources(WEB_SOURCES_PATH)

    with st.expander(f"Menaxho ({len(web_sources)} burime aktive)"):

        # Quick-add from known sources
        st.markdown("**Shto burim të njohur:**")
        known_labels = [s["label"] for s in KNOWN_SOURCES]
        active_urls  = {s["url"] for s in web_sources}

        cols = st.columns(2)
        for i, src in enumerate(KNOWN_SOURCES):
            col = cols[i % 2]
            already = src["url"] in active_urls
            label   = f"{'✅' if already else '➕'} {src['label']}"
            if col.button(label, key=f"ks_{i}", use_container_width=True, disabled=already):
                web_sources.append({"url": src["url"], "label": src["label"], "category": src["category"]})
                save_web_sources(web_sources, WEB_SOURCES_PATH)
                st.cache_resource.clear()
                st.rerun()

        st.markdown("**Ose shto URL të personalizuar:**")
        with st.form("add_web_src", clear_on_submit=True):
            new_url   = st.text_input("URL", placeholder="https://www.example.com/page")
            new_label = st.text_input("Emri", placeholder="p.sh. ZRRE – Tarifat 2024")
            new_cat   = st.selectbox("Kategoria", ["KOSTT", "ZRRE", "KEK", "KESCO", "KEDS",
                                                   "ENTSO-E", "Energy Community",
                                                   "Gazeta Zyrtare", "Tjetër"])
            if st.form_submit_button("➕ Shto", type="primary", use_container_width=True):
                if new_url.startswith("http"):
                    web_sources.append({"url": new_url.strip(),
                                        "label": new_label.strip() or new_url.strip(),
                                        "category": new_cat})
                    save_web_sources(web_sources, WEB_SOURCES_PATH)
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error("URL duhet të fillojë me https://")

        # List + delete active sources
        if web_sources:
            st.markdown("**Burimet aktive:**")
            for i, src in enumerate(web_sources):
                c1, c2 = st.columns([7, 1])
                c1.markdown(
                    f"<span style='font-size:0.82rem'>🌐 <b>{src['label']}</b>"
                    f"<br><span style='font-size:0.72rem;color:#888'>{src['url'][:50]}{'…' if len(src['url'])>50 else ''}</span></span>",
                    unsafe_allow_html=True,
                )
                if c2.button("🗑️", key=f"del_ws_{i}", help="Hiq"):
                    web_sources.pop(i)
                    save_web_sources(web_sources, WEB_SOURCES_PATH)
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
    st.caption("Energy Law & Legal AI · by ENDRIT")


# ── Eager index pre-load ──────────────────────────────────────────────────────
# Must run BEFORE chat renders. When load_index() is called inside
# st.chat_message() the cache_resource spinner can trigger a Streamlit
# re-render that wipes session_state, causing the "home page" symptom.
# Calling it here (top level) ensures a cache hit inside the question handler.
_preload_err: str | None = None
try:
    _col_pre, _built_pre = load_index()
    if "_index_built_at" not in st.session_state:
        st.session_state["_index_built_at"] = _built_pre
except Exception as _exc:
    _preload_err = str(_exc)

if _preload_err:
    st.error(f"❌ Gabim gjatë ngarkimit të indeksit: {_preload_err}")


# ── CHAT ──────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "edit_idx" not in st.session_state:
    st.session_state.edit_idx = None

if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center; padding: 80px 0 40px 0; color: #888;'>
        <div style='font-size:2.5rem'>⚖️</div>
        <div style='font-size:1.3rem; font-weight:600; color:#1a3a5c; margin:10px 0'>Energy Law & Legal AI by ENDRIT</div>
        <div style='font-size:0.9rem'>AI for Energy Law & Regulatory Insight<br>Built on Kosovo & EU Energy Regulatory Standards</div>
    </div>
    """, unsafe_allow_html=True)

# ── HISTORY ───────────────────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            if st.button("✏️ Edito", key=f"edit_{i}",
                         help="Edito këtë pyetje"):
                st.session_state.edit_idx = i
                st.rerun()
        else:
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Burimet ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-card">📄 <b>{src["doc"]}</b> · '
                            f'{src["category"]} · Faqja {src["page"]}'
                            f' <span class="score-badge">{src["score"]}</span><br>'
                            f'<i>{src["snippet"]}...</i></div>',
                            unsafe_allow_html=True,
                        )

# ── EDIT MODE ─────────────────────────────────────────────────────────────────

if st.session_state.edit_idx is not None:
    idx      = st.session_state.edit_idx
    original = st.session_state.messages[idx]["content"]
    edited   = st.text_area("✏️ Edito pyetjen:", value=original,
                             key="edit_textarea", height=80, label_visibility="collapsed")
    c1, c2 = st.columns([3, 1])
    if c1.button("▶ Dërgo", type="primary", use_container_width=True):
        st.session_state.messages = st.session_state.messages[:idx]
        st.session_state.edit_idx = None
        st.session_state["_pending_q"] = edited
        st.rerun()
    if c2.button("✕ Anulo", use_container_width=True):
        st.session_state.edit_idx = None
        st.rerun()
    question = None
else:
    question = st.chat_input("Shkruani pyetjen tuaj juridike...")

# ── QUESTION HANDLING ─────────────────────────────────────────────────────────

if "_pending_q" in st.session_state:
    question = st.session_state.pop("_pending_q")

if question:
    api_key = get_api_key()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            msg = "⚠️ Vendosni API Key (Anthropic) në sidebar para se të vazhdoni."
            st.warning(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg,
                                               "sources": [], "src_type": "none"})
        elif (sum(count_pdfs(LAWS_DIR).values()) == 0
              and not load_web_sources(WEB_SOURCES_PATH)):
            msg = "⚠️ Ngarkoni dokumente ligjore ose shtoni burime web nga sidebar."
            st.warning(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg,
                                               "sources": [], "src_type": "none"})
        else:
            with st.spinner("Duke kërkuar..."):
                answer  = None
                sources = []
                src_type = "none"
                try:
                    index, built_at = load_index()   # instant — already cached above
                    st.session_state["_index_built_at"] = built_at

                    chunks = retrieve(index, question, api_key, top_k=6)
                    answer, sources, src_type = generate_answer(question, chunks, api_key)

                except Exception as exc:
                    st.error(f"❌ Gabim: {exc}")
                    answer = f"❌ Gabim gjatë përpunimit: {exc}"
                    src_type = "none"

                if answer:
                    if src_type == "general":
                        st.info("ℹ️ Nuk u gjet informacion i mjaftueshëm në dokumentet e ngarkuara — përgjigja bazohet në njohuritë e Claude.")
                    elif src_type == "documents":
                        st.success("📄 U gjet në dokumentet e ngarkuara.")

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📚 Burimet ({len(sources)})"):
                            for src in sources:
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
