import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from wordcloud import WordCloud

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Cluster-Enhanced Article Recommendation", layout="wide")

st.title("📚 Cluster-Enhanced Article Recommendation System")
st.markdown("Hybrid Model: TF-IDF + Cosine Similarity + K-Means Clustering")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("News Dataset.csv")
    df['content'] = df['content'].fillna('').str.lower()
    df['title'] = df['title'].fillna('No Title')
    if 'category' in df.columns:
        df['category'] = df['category'].fillna('Unknown')
    return df

df = load_data()

# -------------------------------------------------
# TF-IDF & Similarity
# -------------------------------------------------
@st.cache_resource
def build_model(data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(data['content'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # KMeans Clustering
    num_clusters = min(5, len(data))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    data['cluster'] = clusters

    # Cluster Similarity Matrix
    cluster_sim = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            if data.loc[i, 'cluster'] == data.loc[j, 'cluster']:
                cluster_sim[i][j] = 1

    # Hybrid Similarity
    alpha = 0.6
    beta = 0.4
    hybrid_sim = alpha * cosine_sim + beta * cluster_sim

    return hybrid_sim

hybrid_sim = build_model(df)

# -------------------------------------------------
# Article Selection
# -------------------------------------------------
article = st.selectbox("Select an Article", df['title'])
idx = df[df['title'] == article].index[0]

# -------------------------------------------------
# Recommendations
# -------------------------------------------------
scores = sorted(
    list(enumerate(hybrid_sim[idx])),
    key=lambda x: x[1],
    reverse=True
)[1:6]

st.subheader("🔎 Top 5 Recommendations")

for i, score in scores:
    st.write(f"**{df.loc[i,'title']}**  |  Similarity Score: {score:.2f}")

# -------------------------------------------------
# Word Cloud
# -------------------------------------------------
st.subheader("☁️ Word Cloud of Recommended Articles")

top_content = " ".join([df.loc[i,'content'] for i,_ in scores])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(top_content)

fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")

st.pyplot(fig)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown("Developed by Punam Kanungoe | Hybrid Recommendation Model")
