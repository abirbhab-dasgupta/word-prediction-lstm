# Next Word Prediction with LSTM

A deep learning natural language processing (NLP) application that predicts the most probable next word(s) given a seed phrase or sentence prefix, built with **TensorFlow/Keras** and served via an interactive **Streamlit** web interface.

---

## Architecture Overview

- **Pipeline:** Text tokenization $\rightarrow$ Pre-padding $\rightarrow$ Embedding layer $\rightarrow$ Long Short-Term Memory (LSTM) $\rightarrow$ Dense Softmax classifier.
- **Dataset:** Quotes dataset (`qoute_dataset.csv`) consisting of ~3,000 literary and philosophical quotes.
- **Features:**
  - **Single Next-Word Prediction:** Displays top-$k$ candidate words ranked by posterior probabilities.
  - **Multi-Word Extension:** Generates extended sentence continuations sequentially.
  - **Interactive Interface:** Fast inference with cached model weights, interactive starter prompts, and out-of-vocabulary (OOV) token detection.

---

## Project Structure

```text
├── app.py              # Streamlit interactive web application
├── code.ipynb          # Model training, experimentation, and evaluation notebook
├── lstm_model.h5       # Pre-trained LSTM sequence model weights
├── rnn_model.h5        # Baseline SimpleRNN model weights
├── tokenizer.pkl       # Fitted Keras Tokenizer instance
├── max_len.pkl         # Maximum sequence length configuration
├── qoute_dataset.csv   # Training dataset
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## Quickstart & Setup

### Prerequisites
- **Python:** `3.10` to `3.12` *(Python 3.14 is currently unsupported by TensorFlow)*
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/abirbhab-dasgupta/word-prediction-lstm.git
cd word-prediction-lstm
```

### 2. Create and Activate a Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv nextword_env
  .\nextword_env\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv nextword_env
  source nextword_env/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Once started, the application will be accessible at `http://localhost:8501`.

---

## Model Evaluation Summary

| Architecture | Parameters | Target Metric | Use Case |
| :--- | :--- | :--- | :--- |
| **SimpleRNN** | Embedding + SimpleRNN + Dense | Baseline validation | Comparison benchmark |
| **LSTM** | Embedding + LSTM + Dense | Categorical Crossentropy | Production inference (`app.py`) |

---

