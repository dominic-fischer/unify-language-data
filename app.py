from __future__ import annotations

import streamlit as st

from config import init_page
from app_ui.grammar_tab import render_grammar_tab
from app_ui.vocab_tab import render_vocab_tab

init_page()
st.title("Linguistics Data Browser")

page = st.radio("View", ["Grammar", "Vocab"], horizontal=True, key="page")

if page == "Grammar":
    render_grammar_tab()
else:
    render_vocab_tab()

st.divider()
st.caption("Tip: if you change parsing logic, clear Streamlit cache (⋮ menu → Clear cache).")
