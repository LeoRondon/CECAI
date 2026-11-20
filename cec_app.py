"""Streamlit chat UI for the CEC Assistant.

Run with:  streamlit run cec_app.py
"""

from __future__ import annotations

import streamlit as st

from cec import ollama
from cec.config import settings
from cec.prompts import build_prompt
from cec.retrieval import Match, RuleIndex

EXAMPLE_QUESTIONS = [
    "What does 'bonding' mean?",
    "Do I need a permit before starting electrical work?",
    "What is the minimum conductor size?",
    "Calculated load for a single dwelling",
]

st.set_page_config(page_title="CEC Assistant", page_icon="⚡", layout="centered")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 52rem; }
      .cec-hero h1 { font-size: 2.1rem; margin-bottom: .25rem; }
      .cec-hero p { opacity: .75; margin-top: 0; }
      .cec-score {
        font-size: .75rem; padding: .1rem .5rem; border-radius: 999px;
        background: rgba(245, 158, 11, .15); color: inherit; white-space: nowrap;
      }
      div[data-testid="stExpander"] details summary p { font-size: .92rem; }

      /* Keep LLM output inside the window: wrap long lines instead of overflowing. */
      [data-testid="stChatMessage"],
      [data-testid="stChatMessageContent"] { min-width: 0; max-width: 100%; }
      [data-testid="stChatMessageContent"] {
        overflow-wrap: anywhere; word-break: break-word;
      }
      [data-testid="stChatMessageContent"] pre,
      [data-testid="stChatMessageContent"] pre code {
        white-space: pre-wrap !important; overflow-wrap: anywhere; word-break: break-word;
      }
      [data-testid="stChatMessageContent"] table {
        display: block; max-width: 100%; overflow-x: auto;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading CEC rules…")
def load_index() -> RuleIndex:
    return RuleIndex.load()


def match_to_dict(match: Match) -> dict:
    return {
        "rule": match.rule.rule,
        "title": match.rule.title,
        "text": match.rule.text,
        "score": match.score,
    }


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources · {len(sources)} CEC rules", expanded=False):
        for src in sources:
            st.markdown(
                f"**{src['rule']}** — {src['title']} "
                f"<span class='cec-score'>{src['score']:.0%} match</span>",
                unsafe_allow_html=True,
            )
            st.code(src["text"], language=None, wrap_lines=True)


def render_assistant(message: dict) -> None:
    if message.get("reasoning"):
        with st.expander("🧠 Reasoning", expanded=False):
            st.markdown(message["reasoning"])
    if message.get("error"):
        st.error(message["error"], icon="⚠️")
    else:
        st.markdown(message["content"])
    render_sources(message.get("sources", []))


def answer(question: str, model: str, top_k: int, show_reasoning: bool) -> dict:
    """Retrieve rules, stream the answer into the page, and return the message."""
    message: dict = {"role": "assistant", "content": "", "reasoning": "", "sources": []}
    try:
        with st.spinner("Searching the CEC…"):
            matches = load_index().search(question, top_k)
        message["sources"] = [match_to_dict(m) for m in matches]

        reasoning_box = st.empty()
        answer_box = st.empty()
        answer_box.markdown("_Preparing answer…_")
        prompt = build_prompt(question, matches)

        for chunk in ollama.generate_stream(prompt, model, think=show_reasoning):
            if chunk.kind == "thinking":
                message["reasoning"] += chunk.text
                if show_reasoning:
                    with reasoning_box.expander("🧠 Reasoning…", expanded=True):
                        st.markdown(message["reasoning"])
            else:
                message["content"] += chunk.text
                answer_box.markdown(message["content"] + " ▌")

        message["content"] = message["content"].strip()
        message["reasoning"] = message["reasoning"].strip() if show_reasoning else ""
        reasoning_box.empty()
        answer_box.empty()
    except ollama.OllamaError as exc:
        message["error"] = str(exc)
    return message


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.text_input("Chat model", value=settings.chat_model)
    top_k = st.slider("Rules to retrieve", min_value=1, max_value=15, value=settings.top_k)
    show_reasoning = st.toggle(
        "Show model reasoning",
        value=False,
        help="Slower. Displays the model's thinking (for models that support it).",
    )
    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Try asking")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"side-{q}", width="stretch"):
            st.session_state.pending = q

    st.divider()
    st.caption(
        f"Answers come only from the CEC rule text · Ollama at `{settings.ollama_host}`. "
        "Always verify against the official Code."
    )

# ── Main ───────────────────────────────────────────────────────────────────
st.session_state.setdefault("messages", [])

st.markdown(
    "<div class='cec-hero'><h1>⚡ CEC Assistant</h1>"
    "<p>Ask a question about the Canadian Electrical Code. Answers cite the rules they're based on.</p></div>",
    unsafe_allow_html=True,
)

if not st.session_state.messages and "pending" not in st.session_state:
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(f"💬 {q}", key=f"main-{q}", width="stretch"):
            st.session_state.pending = q
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🔧" if msg["role"] == "user" else "⚡"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg)

question = st.chat_input("Ask about the CEC…") or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🔧"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="⚡"):
        reply = answer(question, model, top_k, show_reasoning)
        render_assistant(reply)
    st.session_state.messages.append(reply)
