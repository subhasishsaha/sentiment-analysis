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

load_dotenv()

# This check is not provided in your code but is good practice
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please add it to your .env file.")
    st.stop()
else:
    genai.configure(api_key=GEMINI_API_KEY)


st.set_page_config(page_title="Play Store Review Analyzer", page_icon="📱", layout="centered")

GOOGLE_DRIVE_FILE_IDS = {
    "multilabel_binarizer": "1eQxx38vR2YHU7cUB8hp-iVZJf0S_AuMN",
    "ensemble_model": "1ZWICtGgeyo4SOJGGzc7EYD4A7x-JKU60",
    "tfidf": "1_dJAeRBxgXWlkf66meNPxojd5filhT-E",
    "sentiment_model": "1QpyEew6f-mUuMpR9Wn0r1zeyCUgNiJ7Q"
}

# --- Model Loading ---
@st.cache_resource
def download_file_from_google_drive(file_id, output_path):
    """Downloads a file from Google Drive."""
    if not os.path.exists(output_path):
        with st.spinner(f"Downloading {os.path.basename(output_path)}..."):
            gdown.download(id=file_id, output=output_path, quiet=False)

@st.cache_resource
def load_resources():
    """Downloads and loads all necessary model files."""
    try:
        # Create directories if they don't exist
        os.makedirs("Models/ensemble_models", exist_ok=True)
        os.makedirs("Models/roberta_tokenizer", exist_ok=True)
        os.makedirs("Models/sentiment_model", exist_ok=True)

        # Download model files
        download_file_from_google_drive(GOOGLE_DRIVE_FILE_IDS["multilabel_binarizer"], "Models/ensemble_models/multilabel_binarizer.pkl")
        download_file_from_google_drive(GOOGLE_DRIVE_FILE_IDS["ensemble_model"], "Models/ensemble_models/ensemble_model.pkl")
        download_file_from_google_drive(GOOGLE_DRIVE_FILE_IDS["tfidf"], "Models/ensemble_models/tfidf.pkl")
        download_file_from_google_drive(GOOGLE_DRIVE_FILE_IDS["sentiment_model"], "Models/sentiment_model/best_model_state.bin")


        mlb = joblib.load("Models/ensemble_models/multilabel_binarizer.pkl")
        ensemble = joblib.load("Models/ensemble_models/ensemble_model.pkl")
        tfidf = joblib.load("Models/ensemble_models/tfidf.pkl")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        class Sentiment_Classifier(nn.Module):
            def __init__(self, n_classes):
                super(Sentiment_Classifier, self).__init__()
                self.roberta = AutoModel.from_pretrained("roberta-base")
                
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
                output = self.drop(pooled_output )
                output = self.out(output)
                return output
    
        model = Sentiment_Classifier(n_classes = 2)


        model.load_state_dict(torch.load("Models/sentiment_model/best_model_state.bin", map_location="cpu"))
        model.eval()
        
        return mlb, ensemble, tfidf, tokenizer, model
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()


try:
    mlb, ensemble, tfidf, tokenizer, model = load_resources()
except Exception as e:
    st.error(f"An error occurred during app initialization: {e}")
    st.stop()
    
softmax = nn.Softmax(dim=1)

# === Text Processing ===
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

def get_labels_batch(texts):
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []
    X = tfidf.transform(texts)
    y_pred = ensemble.predict(X)
    y_pred = np.array(y_pred)
    all_labels = mlb.inverse_transform(y_pred)
    return [labels if labels else ("unknown",) for labels in all_labels]

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
            sentiments.extend(["Positive" if p == 1 else "Negative" for p in torch.argmax(probs, dim=1).tolist()])
    return sentiments

def prepare_label_to_reviews(df, max_reviews_per_label=10):
    label_to_reviews = defaultdict(list)
    for _, row in df[df["sentiment"] == "Negative"].iterrows(): 
        labels = row.get("labels", [])
        if isinstance(labels, str):
            try:
                labels = ast.literal_eval(labels)
            except (ValueError, SyntaxError):
                labels = []
        for label in labels:
            clean = label.strip().lower()
            if clean != "unknown":
                label_to_reviews[clean].append(row["content"])
    return {k: list(dict.fromkeys(v))[:max_reviews_per_label] for k, v in label_to_reviews.items()}

def generate_gemini_suggestions(label_to_reviews, model_name="gemini-1.5-flash", max_reviews=3):
    model_gemini = genai.GenerativeModel(model_name)
    suggestions = {}
    for label, reviews in label_to_reviews.items():
        if not reviews:
            continue
        prompt = f"""You are an AI assistant. Based on the following negative reviews about the category '{label}', suggest actionable improvements:\n\n{chr(10).join('- ' + r for r in reviews[:max_reviews])}"""
        try:
            response = model_gemini.generate_content(prompt)
            suggestions[label] = response.text.strip()
        except Exception as e:
            suggestions[label] = f"⚠️ Error generating suggestions: {e}"
    return suggestions

# === Visualization (Updated with Plotly) ===
def display_visualizations(df):
    st.subheader("Visualizations")
    viz_option = st.selectbox("🧭 Select a Visualization Type", 
                                 ["Label Distribution", "Sentiment Distribution", "Sentiment by Category", "Sentiment by App Version", "Word Cloud"])

    st.markdown("---")

    if viz_option == "Label Distribution":
        st.subheader("📊 Label Distribution")
        all_labels = df["labels"].explode().dropna()
        if not all_labels.empty:
            label_counts = all_labels.value_counts().reset_index()
            label_counts.columns = ['Label', 'Count']
            fig = px.bar(label_counts, 
                         x='Count', 
                         y='Label', 
                         orientation='h',
                         title="Distribution of Review Labels",
                         color='Count',
                         color_continuous_scale=px.colors.sequential.Magma,
                         labels={'Label': 'Category', 'Count': 'Number of Reviews'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No labels to display.")

    elif viz_option == "Sentiment Distribution":
        st.subheader("💬 Overall Sentiment Distribution")
        if not df["sentiment"].empty:
            sentiment_counts = df['sentiment'].value_counts()
            fig = px.pie(values=sentiment_counts.values, 
                         names=sentiment_counts.index,
                         title="Overall Distribution of Sentiments",
                         color_discrete_map={'Positive': 'mediumseagreen', 'Negative': 'indianred'},
                         hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sentiment data to display.")

    elif viz_option == "Sentiment by Category":
        st.subheader("📊 Sentiment Distribution per Category")
        exploded_df = df.explode('labels')
        unique_labels = sorted(exploded_df[exploded_df['labels'] != 'unknown']['labels'].unique())

        if not unique_labels:
            st.info("No specific categories were identified in the reviews to analyze.")
        else:
            selected_label = st.selectbox("Choose a category to analyze:", options=unique_labels)
            
            if selected_label:
                label_df = exploded_df[exploded_df['labels'] == selected_label]
                sentiment_counts = label_df['sentiment'].value_counts()
                
                if not sentiment_counts.empty:
                    fig = px.pie(values=sentiment_counts.values, 
                                 names=sentiment_counts.index,
                                 title=f"Sentiments for '{selected_label.title()}'",
                                 hole=0.3,
                                 color_discrete_map={'Positive': 'mediumseagreen', 'Negative': 'indianred'})
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No sentiment data to display for this category.")

    elif viz_option == "Sentiment by App Version":
        st.subheader("📊 Sentiment Distribution by App Version")
        if 'appVersion' in df.columns and not df['appVersion'].isnull().all():
            unique_versions = sorted(df['appVersion'].dropna().unique())

            if not unique_versions:
                st.info("No specific app versions were found in the data.")
            else:
                selected_version = st.selectbox("Choose an app version to analyze:", options=unique_versions)
                
                if selected_version:
                    version_df = df[df['appVersion'] == selected_version]
                    sentiment_counts = version_df['sentiment'].value_counts()
                    
                    if not sentiment_counts.empty:
                        fig = px.pie(values=sentiment_counts.values, 
                                     names=sentiment_counts.index,
                                     title=f"Sentiments for Version '{selected_version}'",
                                     hole=0.3,
                                     color_discrete_map={'Positive': 'mediumseagreen', 'Negative': 'indianred'})
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write(f"No sentiment data to display for version {selected_version}.")
        else:
            st.info("App version data is not available in the fetched reviews.")

    elif viz_option == "Word Cloud":
        st.subheader("☁️ Word Cloud")
        sentiment = st.selectbox("Choose Sentiment", ["Positive", "Negative"])
        text_data = df[df["sentiment"] == sentiment]["content"].dropna()
        if not text_data.empty:
            text = " ".join(text_data.tolist())
            with st.spinner("Generating word cloud..."):
                wordcloud = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
        else:
            st.info(f"No text available for {sentiment} sentiment.")

# === Streamlit App ===
def main():

    st.title("Play Store Review Analyzer 📱")

    if 'review_df' not in st.session_state:
        st.session_state['review_df'] = pd.DataFrame()

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

            try:
                with st.spinner("Fetching reviews... This might take a moment."):
                    sort_map = {'Newest': Sort.NEWEST, 'Most Relevant': Sort.MOST_RELEVANT}
                    result, _ = reviews(app_id, sort=sort_map[sort_order], count=count, lang="en", country="us")
                    if not result:
                        st.warning("No reviews fetched. The app may have no reviews or the ID could be incorrect.")
                        return
                    df = pd.DataFrame(result)
                
                with st.spinner("Analyzing content..."):
                    df_processed = df.copy()
                    df_processed["content_processed"] = df_processed["content"].apply(preprocess)
                    df_processed.dropna(subset=['content_processed'], inplace=True)
                    
                    # Filter for English content after initial preprocessing
                    if not df_processed.empty:
                        df_processed["content_processed"] = df_processed["content_processed"].apply(filter_english_sentences)
                        df_processed.dropna(subset=['content_processed'], inplace=True)

                    if df_processed.empty:
                        st.warning("No valid English content found after preprocessing.")
                        st.session_state['review_df'] = pd.DataFrame() # Clear old results
                        return

                    texts = df_processed["content_processed"].tolist()
                    df_processed["labels"] = get_labels_batch(texts)
                    df_processed["sentiment"] = predict_sentiments(texts)
                    
                    # Merge results back to the original df to keep all columns
                    # Use 'reviewId' as the key for a robust merge
                    df = df.merge(df_processed[['reviewId', 'sentiment', 'labels']], on='reviewId', how='right')
                    df.reset_index(drop=True, inplace=True)
                
                st.success("✅ Analysis Complete!")
                st.session_state['review_df'] = df
            
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

    with col2:
        if not st.session_state.get('review_df', pd.DataFrame()).empty:
            df_display = st.session_state['review_df']
            
            st.subheader("📊 Analysis Results")
            st.dataframe(df_display[['userName', 'content', 'at', 'appVersion', 'sentiment', 'labels']], height=300, use_container_width=True)

            st.markdown("---")
            
            st.subheader("💡 Gemini Suggestions (for Negative Reviews)")
            with st.spinner("Generating suggestions..."):
                label_to_reviews = prepare_label_to_reviews(df_display)
                if label_to_reviews:
                    suggestions = generate_gemini_suggestions(label_to_reviews)
                    if suggestions:
                        for label, suggestion in suggestions.items():
                            with st.expander(f"Suggestions for: **{label.title()}**"):
                                st.markdown(suggestion)
                    else:
                        st.info("Could not generate suggestions from the negative reviews.")
                else:
                    st.info("No actionable negative reviews found to generate suggestions.")
        else:
            st.info("⬅️ Enter an App ID and click 'Fetch & Analyze Reviews' to begin.")

    st.markdown("<hr>", unsafe_allow_html=True)
    if not st.session_state.get('review_df', pd.DataFrame()).empty:
        display_visualizations(st.session_state['review_df'])
    else:
        st.info("Run an analysis to see visualizations here.")


if __name__ == "__main__":
    main()