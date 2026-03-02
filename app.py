# ==============================================
# Hybrid Content-Based Article Recommendation
# Author: Punam Kanungoe
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('News Dataset.csv') 
    df['content'] = df['content'].fillna('').str.lower()
    df['title'] = df['title'].fillna('No Title')
    if 'category' in df.columns:
        df['category'] = df['category'].fillna('Unknown')
    return df

df = load_data()

# -----------------------------
# Title
# -----------------------------
st.title("Hybrid Content-Based Article Recommendation System")
st.markdown("This system recommends similar news articles using a **Hybrid approach**: Cosine similarity + Clustering.")

# -----------------------------
# TF-IDF Matrix
# -----------------------------
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['content'])

# -----------------------------
# Cosine Similarity
# -----------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -----------------------------
# KMeans Clustering
# -----------------------------
num_clusters = min(5, len(df))
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# -----------------------------
# Hybrid Similarity (Cosine + Cluster)
# -----------------------------
cluster_sim = np.zeros((len(df), len(df)))
for i in range(len(df)):
    for j in range(len(df)):
        if df.loc[i, 'cluster'] == df.loc[j, 'cluster']:
            cluster_sim[i][j] = 1

alpha = 0.6  # weight for cosine similarity
beta = 0.4   # weight for cluster similarity
hybrid_sim = alpha * cosine_sim + beta * cluster_sim

# -----------------------------
# Select Article
# -----------------------------
article = st.selectbox("Select Article:", df['title'])
idx = df[df['title'] == article].index[0]

# -----------------------------
# Top Recommendations
# -----------------------------
scores = sorted(list(enumerate(hybrid_sim[idx])), key=lambda x: x[1], reverse=True)[1:6]

st.subheader("Top Recommendations")
for i, score in scores:
    st.write(f"{df.loc[i,'title']} — Similarity: {score:.2f}")

# -----------------------------
# Word Cloud of Recommendations
# -----------------------------
top_content = " ".join([df.loc[i,'content'] for i,_ in scores])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(top_content)

st.subheader("Word Cloud of Recommended Articles")
st.image(wordcloud.to_array(), use_column_width=True)
