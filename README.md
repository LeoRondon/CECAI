Web app in your browser. Streamlit isn’t installed yet, so install it once first:

py -3.12 -m pip install --user streamlit

Then start the app:

py -3.12 -m streamlit run cec_app.py

It opens at http://localhost:8501. Type a question there, and press Ctrl+C in the terminal to stop it.

Good questions to test the new sections:

“What does ‘bonding’ mean?” (Section 0)
“Do I need a permit before starting electrical work?” (Section 2)
“What is the minimum conductor size?” (Section 4)
“Calculated load for a single dwelling” (Section 8)

Use py -3.12 rather than plain python, because Python 3.12 is the one that now has requests installed. If a script fails with a connection error, start Ollama again.
