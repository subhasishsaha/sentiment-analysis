# Play Store Review Analyzer

An end-to-end NLP and Generative AI platform for analyzing Google Play Store reviews at scale. The application collects app reviews, performs sentiment analysis, identifies issue categories through multi-label classification, and generates actionable improvement recommendations for developers using Google Gemini.

## Overview

Understanding user feedback is critical for improving mobile applications. This project automates the review analysis process by combining:

- Review scraping from Google Play Store
- Advanced text preprocessing
- Deep Learning-based sentiment analysis
- Multi-label issue classification
- Interactive visual analytics
- AI-generated improvement recommendations

The result is a dashboard that helps developers quickly identify recurring issues, monitor user sentiment, and prioritize product improvements.

---

## Key Features

### Review Collection

- Fetch reviews directly from Google Play Store using App IDs.
- Support for configurable review counts.
- Analyze reviews from any publicly available Android application.

### Text Processing

- URL and emoji removal.
- Text normalization and cleaning.
- Filtering of noisy review content.

### Sentiment Analysis

- Fine-tuned RoBERTa model.
- Binary sentiment classification:
  - Positive
  - Negative

### Multi-Label Issue Detection

Classifies reviews into multiple categories simultaneously, such as:

- Bugs
- Performance
- UI/UX
- Ads
- Cost
- Features
- Account Issues
- Updates

The classification pipeline uses an ensemble approach combining:

- Logistic Regression
- Random Forest
- XGBoost

### AI-Powered Recommendations

Negative reviews are grouped by issue category and summarized using Google Gemini to generate:

- Improvement suggestions
- Product enhancement recommendations
- Priority issue summaries

### Interactive Visualizations

- Sentiment distribution
- Label distribution
- Category-wise sentiment analysis
- Version-wise sentiment tracking
- Word clouds
- Review exploration dashboard

---

## System Architecture

```text
Google Play Reviews
         |
         v
  Text Preprocessing
         |
         +----------------+
         |                |
         v                v
 Sentiment Model    Label Classifier
   (RoBERTa)         (Ensemble ML)
         |                |
         +--------+-------+
                  |
                  v
      Analytics & Insights
                  |
                  v
     Gemini Recommendation Engine
                  |
                  v
        Streamlit Dashboard
```

---

## Repository Structure

```text
playstore-review-analyzer/
│
├── Dataset/
│   └── gpreviews_2.csv
│
├── Test Notebooks/
│   ├── binary-sentiment-classification-using-roberta.ipynb
│   ├── initial-multi-labelling-using-bart-large.ipynb
│   ├── multi-label-classification-using-ensemble-approach.ipynb
│   └── suggestion-using-gemini.ipynb
│
├── Models/
│   ├── sentiment_model/
│   ├── roberta_tokenizer/
│   └── ensemble_models/
│
├── app.py
├── update_local_model.py
├── requirements.txt
└── README.md
```

---

## Technology Stack

### Machine Learning

- Scikit-learn
- XGBoost
- TF-IDF Vectorization

### Deep Learning

- PyTorch
- RoBERTa
- Transformers

### Generative AI

- Google Gemini

### Data Collection

- Google Play Scraper

### Visualization

- Plotly
- WordCloud
- Pandas

### Application Framework

- Streamlit

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/subhasishsaha/playstore-review-analyzer.git
cd playstore-review-analyzer
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## Models

The repository contains the trained models required to run the application.

If you wish to retrain the models:

### Sentiment Model

Run:

```text
binary-sentiment-classification-using-roberta.ipynb
```

Outputs:

- best_model_state.bin
- RoBERTa tokenizer files

### Multi-Label Classifier

Run:

```text
multi-label-classification-using-ensemble-approach.ipynb
```

Outputs:

- ensemble_model.pkl
- tfidf.pkl
- multilabel_binarizer.pkl

---

## Running the Application

```bash
streamlit run app.py
```

The application will launch in your browser automatically.

---

## Methodology

### 1. Dataset Preparation

The original review dataset was automatically labeled using:

- BART Large MNLI
- Zero-Shot Classification

This created training labels for issue categories.

### 2. Multi-Label Classification

A TF-IDF representation is generated from review text and passed to an ensemble classifier composed of:

- Logistic Regression
- Random Forest
- XGBoost

### 3. Sentiment Analysis

A fine-tuned RoBERTa model predicts review sentiment.

### 4. Recommendation Generation

Negative reviews are grouped by category and passed to Gemini for summarization and recommendation generation.

---

## Usage

### Step 1

Enter a Google Play Store App ID.

Example:

```text
com.whatsapp
```

### Step 2

Select:

- Review type
- Number of reviews

### Step 3

Click:

```text
Fetch & Analyze
```

### Step 4

Explore:

- Review table
- Sentiment predictions
- Category labels
- Visual analytics
- AI-generated recommendations

---

## Future Improvements

- Neutral sentiment detection.
- Multilingual review analysis.
- Aspect-based sentiment analysis.
- Real-time monitoring dashboard.
- Trend detection across app versions.
- Comparative analysis between competing apps.
- Cloud deployment and automated model updates.

---

## Disclaimer

This project is intended for educational and research purposes. Generated recommendations should be reviewed before being used for product or business decisions.

---

## License

This project is released under the repository's LICENSE file.
