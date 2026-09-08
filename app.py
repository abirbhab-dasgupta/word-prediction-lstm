import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.utils import pad_sequences
from tensorflow.keras.models import load_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Next Word Predictor | LSTM",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        background: rgba(79, 70, 229, 0.08);
        border: 1px solid rgba(79, 70, 229, 0.25);
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 15px;
    }
    .highlight-word {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4F46E5;
        background: rgba(79, 70, 229, 0.12);
        padding: 4px 12px;
        border-radius: 8px;
        display: inline-block;
    }
    .chip {
        display: inline-block;
        padding: 4px 12px;
        margin: 4px 4px 4px 0px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
        background: #F1F5F9;
        color: #334155;
    }
    </style>
""", unsafe_allow_html=True)

# --- Resource Loading with Caching ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner="Loading LSTM model...")
def load_lstm_model():
    model_path = os.path.join(BASE_DIR, "lstm_model.h5")
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        return None
    return load_model(model_path, compile=False)

@st.cache_resource(show_spinner="Loading Tokenizer and configurations...")
def load_tokenizer_and_config():
    tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")
    maxlen_path = os.path.join(BASE_DIR, "max_len.pkl")

    if not os.path.exists(tokenizer_path):
        st.error(f"Tokenizer file '{tokenizer_path}' not found.")
        return None, None, None

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Reconstruct index-to-word map
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}

    # Load max_len if available, otherwise default to 745 from training
    max_len = 745
    if os.path.exists(maxlen_path):
        with open(maxlen_path, "rb") as f:
            max_len = pickle.load(f)

    return tokenizer, index_to_word, int(max_len)

# Load artifacts
model = load_lstm_model()
tokenizer, index_to_word, max_len = load_tokenizer_and_config()

# --- Prediction Functions ---
def predict_next_words(model, tokenizer, index_to_word, text, max_len, top_k=5):
    """Predicts top-k likely next words and their probabilities."""
    text_clean = text.strip().lower()
    seq = tokenizer.texts_to_sequences([text_clean])
    
    if not seq or not seq[0]:
        return []

    padded_seq = pad_sequences([seq[0]], maxlen=max_len, padding='pre')
    preds = model.predict(padded_seq, verbose=0)[0]

    # Get indices of top_k predictions
    top_indices = np.argsort(preds)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        word = index_to_word.get(idx, None)
        prob = float(preds[idx])
        if word and prob > 0:
            results.append((word, prob))
            
    return results

def generate_sequence(model, tokenizer, index_to_word, seed_text, max_len, n_words=5):
    """Generates multiple sequential next words."""
    current_text = seed_text.strip()
    generated_words = []

    for _ in range(n_words):
        seq = tokenizer.texts_to_sequences([current_text.lower()])
        if not seq or not seq[0]:
            break

        padded = pad_sequences([seq[0]], maxlen=max_len, padding='pre')
        pred = model.predict(padded, verbose=0)[0]
        pred_index = int(np.argmax(pred))
        
        next_word = index_to_word.get(pred_index, "")
        if not next_word or next_word in generated_words[-2:]:
            # Break if empty or repeated cycle
            break

        generated_words.append(next_word)
        current_text += " " + next_word

    return current_text, generated_words

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=120)
    st.markdown("### 🧠 Model Overview")
    st.info(
        "**Architecture:** Embedding + LSTM + Dense (Softmax)\n\n"
        "**Trained on:** Quote Dataset\n\n"
        f"**Vocabulary Size:** {len(tokenizer.word_index) if tokenizer else 'N/A'} words\n\n"
        f"**Max Sequence Length:** {max_len} tokens"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Generation Settings")
    mode = st.radio("Prediction Mode", ["Single Next Word (with Top Candidates)", "Multi-Word Sentence Extension"])
    
    if mode == "Single Next Word (with Top Candidates)":
        top_k = st.slider("Top Candidates to show", min_value=1, max_value=10, value=5)
    else:
        num_words = st.slider("Number of words to generate", min_value=1, max_value=20, value=6)

    st.markdown("---")
    st.caption("Developed with TensorFlow/Keras & Streamlit")

# --- Main UI ---
st.markdown('<div class="main-header">🔮 Next Word Prediction with LSTM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Input a phrase and let the trained Long Short-Term Memory network predict the continuation.</div>', unsafe_allow_html=True)

if model is None or tokenizer is None:
    st.stop()

# Session state for user input
if "text_input_val" not in st.session_state:
    st.session_state["text_input_val"] = "She is"

# Quick prompt chips for user convenience
st.markdown("**Try one of these sample starters:**")
col1, col2, col3, col4 = st.columns(4)

if col1.button("“She is”"):
    st.session_state["text_input_val"] = "She is"
    st.rerun()
if col2.button("“It is our”"):
    st.session_state["text_input_val"] = "It is our"
    st.rerun()
if col3.button("“There are only”"):
    st.session_state["text_input_val"] = "There are only"
    st.rerun()
if col4.button("“The world as”"):
    st.session_state["text_input_val"] = "The world as"
    st.rerun()

# Text input
user_input = st.text_input(
    "Enter a phrase or beginning of a sentence:",
    key="text_input_val",
    placeholder="Type something here... (e.g. Life is full of)"
)

# Execution trigger
predict_button = st.button("✨ Predict Next Word", type="primary", use_container_width=True)

if user_input:
    # Verify words in tokenizer
    input_tokens = user_input.strip().lower().split()
    oov_tokens = [w for w in input_tokens if w not in tokenizer.word_index]

    if oov_tokens:
        st.warning(f"⚠️ Note: The following word(s) are not in the training vocabulary and may be ignored: **{', '.join(oov_tokens)}**")

if predict_button or user_input:
    if not user_input.strip():
        st.warning("Please enter a non-empty phrase.")
    else:
        if mode == "Single Next Word (with Top Candidates)":
            with st.spinner("Analyzing sequence and predicting..."):
                top_predictions = predict_next_words(
                    model, tokenizer, index_to_word, user_input, max_len, top_k=top_k
                )

            if top_predictions:
                best_word, best_prob = top_predictions[0]
                
                # Display best next word
                st.markdown(f"""
                    <div class="prediction-box">
                        <span style="font-size: 1rem; color: #64748B;">Predicted Continuation:</span><br/>
                        <span style="font-size: 1.3rem;">{user_input.strip()}</span> 
                        <span class="highlight-word">{best_word}</span>
                    </div>
                """, unsafe_allow_html=True)

                st.write("")
                st.markdown("#### 📊 Top Candidate Probabilities")
                
                # Display candidate predictions with progress bars
                candidate_cols = st.columns([2, 3])
                with candidate_cols[0]:
                    for rank, (word, prob) in enumerate(top_predictions, 1):
                        st.write(f"**#{rank} `{word}`** — {prob * 100:.2f}%")
                        st.progress(min(float(prob), 1.0))
                
                with candidate_cols[1]:
                    # Quick append buttons for interactive writing
                    st.markdown("**Click any candidate to append to your input:**")
                    for word, prob in top_predictions:
                        if st.button(f"+ {word} ({prob * 100:.1f}%)", key=f"btn_{word}"):
                            st.session_state["text_input_val"] = f"{user_input.strip()} {word}"
                            st.rerun()

            else:
                st.info("No prediction could be generated. Please try words commonly found in quotes or natural text.")

        else:
            # Multi-word sentence generation
            with st.spinner(f"Generating next {num_words} words..."):
                extended_text, generated_tokens = generate_sequence(
                    model, tokenizer, index_to_word, user_input, max_len, n_words=num_words
                )

            st.markdown(f"""
                <div class="prediction-box">
                    <span style="font-size: 1rem; color: #64748B;">Generated Quote Extension:</span><br/>
                    <span style="font-size: 1.3rem;">{user_input.strip()}</span> 
                    <span class="highlight-word">{' '.join(generated_tokens)}</span>
                </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.markdown(f"**Words generated:** {len(generated_tokens)} / {num_words}")
