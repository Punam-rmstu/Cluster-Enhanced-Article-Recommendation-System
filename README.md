# Cluster-Enhanced Content-Based Article Recommendation System

**Author:** Punam Kanungoe
**Tech Stack:** Python, Streamlit, Scikit-learn, Pandas, Matplotlib, WordCloud  
**Approach:** TF-IDF + Cosine Similarity + K-Means Clustering (Hybrid)  

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Methodology](#methodology)  
    - [1️⃣ Data Preprocessing](#1️⃣-data-preprocessing)  
    - [2️⃣ TF-IDF Vectorization](#2️⃣-tf-idf-vectorization)  
    - [3️⃣ Cosine Similarity](#3️⃣-cosine-similarity)  
    - [4️⃣ K-Means Clustering](#4️⃣-k-means-clustering)  
    - [5️⃣ Hybrid Recommendation Model](#5️⃣-hybrid-recommendation-model)  
4. [Model Evaluation](#model-evaluation)  
5. [Web Application](#web-application)  
6. [Installation & Usage](#installation--usage)  
7. [Future Enhancements](#future-enhancements)  
8. [License](#license)  

---

## Project Overview
This project implements a **hybrid content-based article recommendation system** using a combination of **TF-IDF cosine similarity** and **K-Means clustering**. The hybrid approach improves the relevance of recommendations by considering both textual similarity and article clustering.

The system allows users to select an article and get the **top 5 recommended articles**, along with a **word cloud** visualization of the recommended content. Users can dynamically adjust the weights of cosine and cluster similarity.

---

## Dataset
- The dataset used in this project was collected from [Kaggle](https://www.kaggle.com/datasets/khushikhushikhushi/comprehensive-news-articles-dataset/data) with at least the following columns:  
  - `title` – The article title  
  - `content` – Full text of the article  
  - `category` – Optional, for evaluation purposes  

- Sample preview:

| title | content | category |
|-------|---------|----------|
| Article 1 | Lorem ipsum ... | World |
| Article 2 | Dolor sit amet ... | Sports |

---

## Methodology

### 1️⃣ Data Preprocessing
- Fill missing values in `content` with empty strings.  
- Convert all text to lowercase.  
- Optional: Fill missing `title` and `category` fields.  

### 2️⃣ TF-IDF Vectorization
- Convert article content into numerical vectors using **TF-IDF**.  
- Limit features to 5000 for efficiency.  

### 3️⃣ Cosine Similarity
- Compute pairwise **cosine similarity** between articles.  
- Cosine similarity captures textual closeness between articles.  

### 4️⃣ K-Means Clustering
- Articles are clustered using **K-Means** (default 5 clusters).  
- Cluster similarity is 1 if two articles are in the same cluster, otherwise 0.  
- Cluster information is used to enhance recommendations.  

### 5️⃣ Hybrid Recommendation Model
- Combine **cosine similarity** and **cluster similarity** using weights:  

\[
\text{Hybrid Similarity} = \alpha \cdot \text{Cosine} + \beta \cdot \text{Cluster}, \quad \alpha + \beta = 1
\]

- Default: `α = 0.6`, `β = 0.4`  
- Top 5 articles with highest hybrid similarity are recommended.  

---

## Model Evaluation
- Precision@5, Recall@5, and F1 score are computed for cosine similarity, Euclidean similarity, and hybrid model.  
- Hybrid model improves recommendation relevance by leveraging clustering.  

| Method | Precision@5 | Recall@5 | F1 Score |
|--------|------------|----------|----------|
| TF-IDF + Cosine |0.362286 | 0.018297 | 0.034835|
| TF-IDF + Euclidean | 0.362286 | 0.018297 | 0.034835 |
| Hybrid (Cosine + Cluster) |0.379143 |	0.019149 |	0.036456 |

---		


## Web Application
- Built using **Streamlit**.  
- Features:
  - Select an article to get recommendations.  
  - Adjust **cosine (α) and cluster (β) weights** dynamically.  
  - Show **cluster ID** of selected article.  
  - Display **top 5 recommended articles**.  
  - Generate **word cloud** of top recommendations.  
  - Optional display of **model evaluation metrics**.  
  -Live Web App
   You can access the deployed app without running locally:
  🔗 Click here to try the Hybrid Article Recommendation System
   https://cluster-enhanced-article-recommendation-system-lshvdvd8zswibui.streamlit.app/ 
---

## Installation & Usage (Locally)

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/hybrid-article-recommender.git
cd hybrid-article-recommender

## 2. Install Dependencies
```bash
pip install -r requirements.txt

### 3. Run Streamlit Web App
```bash
streamlit run app.py

### 4. Access the App

Local URL: http://localhost:8501

Optional: Deploy on Streamlit Cloud or HuggingFace Spaces.

## Future Enhancements

-Include user-based collaborative filtering for hybrid recommendations.
-Add topic modeling with LDA to further improve relevance.
-Support real-time dataset updates via API.
-Improve visualization with interactive plots.

## License

This project is licensed under MIT License.
