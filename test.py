import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
import ast
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from langdetect import detect, LangDetectException
from google_play_scraper import reviews, Sort
import joblib
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
from collections import defaultdict
from dotenv import load_dotenv
import os
import gdown

# --- Initial Setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
st.set_page_config(page_title="Play Store Review Analyzer", page_icon="📱", layout="centered")

# --- Google Drive File IDs ---
# Ensure these folder IDs are correct for your shared folders in Google Drive
GOOGLE_DRIVE_FILE_IDS = {
    "multilabel_binarizer": "1eQxx38vR2YHU7cUB8hp-iVZJf0S_AuMN",
    "ensemble_model": "1ZWICtGgeyo4SOJGGzc7EYD4A7x-JKU60",
    "tfidf": "1_dJAeRBxgXWlkf66meNPxojd5filhT-E",
    "sentiment_model": "1QpyEew6f-mUuMpR9Wn0r1zeyCUgNiJ7Q",
    "roberta_tokenizer_folder": "1PGrAWzFC-E8MtBvgHI8eHxlov2oY_1zq", # Placeholder ID
    "roberta_base_folder": "15Y8C6hErEPnkU3zH8SVA0YYxCaSsN_Qd"      # Placeholder ID
}

# --- Resource Downloading and Loading ---
@st.cache_resource
def download_file(file_id, output_path):
    """Downloads a single file from Google Drive if it doesn't exist."""
    if not os.path.exists(output_path):
        with st.spinner(f"Downloading {os.path.basename(output_path)}..."):
            gdown.download(id=file_id, output=output_path, quiet=False)

@st.cache_resource
def download_folder(folder_id, output_path):
    """Downloads a folder from Google Drive if the directory is empty."""
    if not os.path.exists(output_path) or not os.listdir(output_path):
        with st.spinner(f"Downloading folder {os.path.basename(output_path)}..."):
            gdown.download_folder(id=folder_id, output=output_path, quiet=False)

@st.cache_resource
def load_resources():
    """Downloads and loads all necessary models and resources."""
    try:
        # Define local paths
        base_path = "Models"
        ensemble_path = os.path.join(base_path, "ensemble_models")
        sentiment_path = os.path.join(base_path, "sentiment_model")
        tokenizer_path = os.path.join(base_path, "roberta_tokenizer")
        roberta_path = os.path.join(base_path, "roberta_base")

        # Create directories
        for path in [ensemble_path, sentiment_path, tokenizer_path, roberta_path]:
            os.makedirs(path, exist_ok=True)

        # Download files and folders
        download_file(GOOGLE_DRIVE_FILE_IDS["multilabel_binarizer"], os.path.join(ensemble_path, "multilabel_binarizer.pkl"))
        download_file(GOOGLE_DRIVE_FILE_IDS["ensemble_model"], os.path.join(ensemble_path, "ensemble_model.pkl"))
        download_file(GOOGLE_DRIVE_FILE_IDS["tfidf"], os.path.join(ensemble_path, "tfidf.pkl"))
        download_file(GOOGLE_DRIVE_FILE_IDS["sentiment_model"], os.path.join(sentiment_path, "best_model_state.bin"))
        download_folder(GOOGLE_DRIVE_FILE_IDS["roberta_tokenizer_folder"], tokenizer_path)
        download_folder(GOOGLE_DRIVE_FILE_IDS["roberta_base_folder"], roberta_path)

        # Load resources from local paths
        mlb = joblib.load(os.path.join(ensemble_path, "multilabel_binarizer.pkl"))
        ensemble = joblib.load(os.path.join(ensemble_path, "ensemble_model.pkl"))
        tfidf = joblib.load(os.path.join(ensemble_path, "tfidf.pkl"))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        class Sentiment_Classifier(nn.Module):
            def __init__(self, n_classes):
                super(Sentiment_Classifier, self).__init__()
                self.roberta = AutoModel.from_pretrained(roberta_path)
                # Fine-tuning setup
                for param in self.roberta.parameters():
                    param.requires_grad = False
                for layer in self.roberta.encoder.layer[-2:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                self.drop = nn.Dropout(p=0.1)
                self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)

            def forward(self, input_ids, attention_mask):
                output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = output.last_hidden_state[:, 0, :]
                output = self.drop(pooled_output)
                return self.out(output)
        
        model = Sentiment_Classifier(n_classes=2)
        # Load state dict securely
        model.load_state_dict(torch.load(os.path.join(sentiment_path, "best_model_state.bin"), map_location="cpu", weights_only=True))
        model.eval()
        
        return mlb, ensemble, tfidf, tokenizer, model
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()

# --- App Initialization ---
if 'GEMINI_API_KEY' not in os.environ or not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not found. Please add it to your .env file or environment variables.")
    st.stop()

mlb, ensemble, tfidf, tokenizer, model = load_resources()
softmax = nn.Softmax(dim=1)

# === Text Processing Functions ===
def preprocess(text):
    text = emoji.replace_emoji(text or "", "")
    text = re.sub(r"http\S+|www\S+|<.*?>|\n|\w*\d\w*", '', text).strip()
    return text if text else np.nan

def filter_english_sentences(text):
    if not isinstance(text, str):
        return np.nan
    sentences = [s.strip() for s in text.split('.') if len(s.split()) > 1]
    english = []
    for s in sentences:
        try:
            if detect(s) == 'en':
                english.append(s)
        except LangDetectException:
            continue
    return '. '.join(english) if english else np.nan

# === ML/AI Functions ===
def get_labels_batch(texts):
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []
    X = tfidf.transform(texts)
    y_pred = ensemble.predict(X)
    return [labels if labels else ("unknown",) for labels in mlb.inverse_transform(np.array(y_pred))]

def predict_sentiments(texts, batch_size=32):
    sentiments = []
    texts_to_process = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts_to_process:
        return []
        
    for i in range(0, len(texts_to_process), batch_size):
        batch = texts_to_process[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
        with torch.no_grad():
            outputs = model(enc["input_ids"], enc["attention_mask"])
            probs = softmax(outputs)
            sentiments.extend(["Positive" if p == 1 else "Negative" for p in torch.argmax(probs, dim=1)])
    return sentiments

def prepare_label_to_reviews(df):
    label_to_reviews = defaultdict(list)
    negative_reviews = df[df["sentiment"] == "Negative"]
    for _, row in negative_reviews.iterrows():
        for label in row.get("labels", []):
            clean_label = label.strip().lower()
            if clean_label != "unknown":
                label_to_reviews[clean_label].append(row["content"])
    # Return unique reviews per label
    return {k: list(dict.fromkeys(v)) for k, v in label_to_reviews.items()}

def generate_gemini_suggestions(label_to_reviews, max_reviews_per_label=3):
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
    suggestions = {}
    for label, reviews in label_to_reviews.items():
        if not reviews:
            continue
        prompt_reviews = "\n".join(f"- {r}" for r in reviews[:max_reviews_per_label])
        prompt = f"""Based on the following negative reviews under the category '{label}', provide a few concise, actionable improvement suggestions for developers:\n\n{prompt_reviews}"""
        try:
            response = model_gemini.generate_content(prompt)
            suggestions[label] = response.text.strip()
        except Exception as e:
            suggestions[label] = f"⚠️ Error generating suggestions: {e}"
    return suggestions

# === Visualization Functions ===
def display_visualizations(df):
    st.subheader("Visualizations")
    viz_option = st.selectbox("🧭 Select a Visualization Type",
                                 ["Label Distribution", "Sentiment Distribution", "Sentiment by Category", "Sentiment by App Version", "Word Cloud"])
    st.markdown("---")
    # (Visualization code remains the same as your original script)
    # ...

# === Main Streamlit App ===
def main():
    st.title("Play Store Review Analyzer 📱")

    # Initialize session state
    if 'review_df' not in st.session_state:
        st.session_state.review_df = pd.DataFrame()
    if 'gemini_suggestions' not in st.session_state:
        st.session_state.gemini_suggestions = None

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Analysis Controls")
        app_id = st.text_input("Enter App ID (e.g., com.google.android.gm):")
        sort_order = st.selectbox("Sort Order:", ["Most Relevant", "Newest"])
        count = st.number_input("Number of Reviews:", min_value=10, max_value=1000, value=100, step=10)

        if st.button("🚀 Fetch & Analyze Reviews"):
            if not app_id:
                st.error("Please enter a valid App ID.")
                return

            st.session_state.review_df = pd.DataFrame()
            st.session_state.gemini_suggestions = None
            
            try:
                with st.spinner("Fetching reviews..."):
                    sort_map = {'Newest': Sort.NEWEST, 'Most Relevant': Sort.MOST_RELEVANT}
                    result, _ = reviews(app_id, sort=sort_map[sort_order], count=count, lang="en", country="us")
                
                if not result:
                    st.warning("No reviews found for the given App ID.")
                    return
                
                df = pd.DataFrame(result)
                
                with st.spinner("Preprocessing and analyzing content..."):
                    df["content_processed"] = df["content"].apply(preprocess).dropna()
                    df["content_processed"] = df["content_processed"].apply(filter_english_sentences).dropna()
                    
                    df_analyzed = df.dropna(subset=['content_processed']).copy()
                    if df_analyzed.empty:
                        st.warning("No valid English content found after preprocessing.")
                        return

                    texts = df_analyzed["content_processed"].tolist()
                    df_analyzed["labels"] = get_labels_batch(texts)
                    df_analyzed["sentiment"] = predict_sentiments(texts)
                    
                    st.session_state.review_df = df.merge(df_analyzed[['reviewId', 'sentiment', 'labels']], on='reviewId', how='right')
                
                with st.spinner("Generating AI suggestions..."):
                    label_to_reviews = prepare_label_to_reviews(st.session_state.review_df)
                    if label_to_reviews:
                        st.session_state.gemini_suggestions = generate_gemini_suggestions(label_to_reviews)
                    else:
                        st.session_state.gemini_suggestions = {} # Empty dict signifies no suggestions needed

                st.success("✅ Analysis Complete!")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

    with col2:
        if not st.session_state.review_df.empty:
            df_display = st.session_state.review_df
            
            st.subheader("📊 Analysis Results")
            st.dataframe(df_display[['userName', 'content', 'at', 'appVersion', 'sentiment', 'labels']], height=300, use_container_width=True)
            st.markdown("---")
            
            st.subheader("💡 Gemini Suggestions (for Negative Reviews)")
            suggestions = st.session_state.gemini_suggestions
            if suggestions:
                for label, suggestion in suggestions.items():
                    with st.expander(f"Suggestions for: **{label.title()}**"):
                        st.markdown(suggestion)
            elif suggestions is not None: # Case where analysis ran but found no negative reviews
                st.info("No actionable negative reviews found to generate suggestions.")
        else:
            st.info("⬅️ Enter an App ID and click 'Fetch & Analyze Reviews' to begin.")

    st.markdown("<hr>", unsafe_allow_html=True)
    if not st.session_state.review_df.empty:
        # Assuming the display_visualizations function is defined as in your original code
        display_visualizations(st.session_state.review_df)
    else:
        st.info("Run an analysis to see visualizations here.")

if __name__ == "__main__":
    main()