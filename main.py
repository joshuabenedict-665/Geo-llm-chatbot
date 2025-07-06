import streamlit as st
from llama_cpp import Llama
from utils.intent_classifier import IntentClassifier  # Your BERT classifier

# --- App Config ---
st.set_page_config(
    page_title="GeoChatbot",
    page_icon="🌍",
    layout="centered"
)

# --- Load Models (Cached) ---
@st.cache_resource
def load_models():
    bert_classifier = IntentClassifier("./models/bert_model")
    llm = Llama(
        model_path="./models/gemma-2-2b-it-Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=8
    )
    return bert_classifier, llm

bert_classifier, llm = load_models()

# --- Intent-Specific Prompts ---
def build_prompt(intent, query):
    if intent == "tool_query":
        return f"""You are a geospatial software expert. Explain concisely:
                Question: {query}
                Answer in 2-3 sentences for a GIS beginner:"""
    elif intent == "district_query":
        return f"""Summarize geospatial details about this location:
                Question: {query}
                Include key stats and data sources in 2 sentences:"""
    else:
        return f"""Answer this general geospatial question:
                Question: {query}
                Be concise and technical:"""

# --- Streamlit UI ---
st.title("🌍 Geospatial Chatbot")
st.caption("Ask about GIS tools, district data, or general concepts")

# Chat input
query = st.chat_input("Type your geospatial question...")

if query:
    with st.spinner("Analyzing..."):
        # Classify intent
        intent_result = bert_classifier.predict(query)
        intent = intent_result["intent"]
        
        # Generate prompt
        prompt = build_prompt(intent, query)
        
        # Stream Gemma's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.3,
                max_tokens=150
            ):
                token = chunk["choices"][0]["delta"].get("content", "")
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
    
    # Debug info (collapse by default)
    with st.expander("🔍 Debug Details"):
        st.json({
            "query": query,
            "intent": intent,
            "confidence": intent_result["confidence"],
            "prompt": prompt
        })