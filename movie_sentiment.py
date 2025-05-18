from textblob import TextBlob
import nltk
import text2emotion as te
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize analyzers
vader_analyzer = SentimentIntensityAnalyzer()

# ✅ TextBlob sentiment
def textBlob(text):
    tb = TextBlob(text)
    polarity = round(tb.polarity, 2)
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# ✅ VADER sentiment
def vader(text):
    scores = vader_analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# ✅ Text2Emotion (Emotion Classifier)
def text2emotion_sentiment(text):
    emotion = dict(te.get_emotion(text))
    if not emotion:
        return "Neutral"
    sorted_emotion = sorted(emotion.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top_emotion = sorted_emotion[0][0]
    if len(sorted_emotion) > 1 and (
        sorted_emotion[1][1] >= 0.5 or sorted_emotion[1][1] == sorted_emotion[0][1]):
        top_emotion += f" - {sorted_emotion[1][0]}"
    return top_emotion

# ✅ Plotting function
def plot_pie_chart(labels, values):
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=[v * 100 / sum(values) for v in values],
            hoverinfo="label+percent",
            textinfo="value"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# ✅ Main analysis logic
def analyze_reviews(reviews, model):
    if model == "TextBlob":
        results = [textBlob(r) for r in reviews]
    elif model == "Vader":
        results = [vader(r) for r in reviews]
    elif model == "Text2emotion":
        results = [text2emotion_sentiment(r) for r in reviews]
    else:
        results = []
    return dict(pd.Series(results).value_counts())

# ✅ Streamlit app
st.set_page_config(page_title="Movie Sentiment Analyzer", layout="centered")
st.title("🎬 Movie Sentiment Analyzer")
st.markdown("Upload or input movie reviews to analyze sentiment or emotion.")

model = st.selectbox("Choose a model", ["TextBlob", "Vader", "Text2emotion"])

st.subheader("Enter Reviews")
reviews_text = st.text_area("Paste reviews here (one per line)", height=200)

st.subheader("Or Upload CSV")
uploaded_file = st.file_uploader("Upload a CSV file with one column of reviews", type=["csv"])

reviews = []
if reviews_text:
    reviews.extend([line.strip() for line in reviews_text.strip().split("\n") if line.strip()])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if len(df.columns) >= 1:
        reviews.extend(df.iloc[:, 0].dropna().astype(str).tolist())
    else:
        st.error("CSV must have at least one column of reviews.")

if reviews:
    st.success(f"{len(reviews)} reviews ready for analysis.")
    result = analyze_reviews(reviews, model)
    st.subheader("Sentiment Results")
    for k, v in result.items():
        st.write(f"**{k}**: {v}")

    st.subheader("Visual Representation")
    plot_pie_chart(list(result.keys()), list(result.values()))
else:
    st.info("Enter reviews or upload a file to begin.")


