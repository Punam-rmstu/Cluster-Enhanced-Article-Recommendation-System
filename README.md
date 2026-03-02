# Cluster-Enhanced Content-Based Article Recommendation System

**Author:** Punam Kanungoe

**Tech Stack:** Python, Streamlit, Scikit-learn, Pandas, Matplotlib, WordCloud  
**Approach:** TF-IDF + Cosine Similarity + K-Means Clustering (Hybrid)  

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [TF-IDF Vectorization](#tfidf-vectorization)
  - [Cosine Similarity](#cosine-similarity)
  - [K-Means Clustering](#k-means-clustering)
  - [Hybrid Recommendation Model](#hybrid-recommendation-model)
- [Model Evaluation](#model-evaluation)
- [Web Application](#web-application)
- [Installation & Usage](#installation--usage)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Project Overview

This project implements a **hybrid content-based article recommendation system** using a combination of **TF-IDF cosine similarity** and **K-Means clustering**. The hybrid approach improves the relevance of recommendations by considering both textual similarity and article clustering patterns.

The system allows users to:
- Select an article and receive the **top 5 recommended articles**
- View a **word cloud** visualization of the recommended content
- Dynamically adjust the weights of cosine and cluster similarity to customize recommendations

---

## Dataset

The dataset used in this project was collected from [Kaggle - Comprehensive News Articles Dataset](https://www.kaggle.com/datasets/khushikhushikhushi/comprehensive-news-articles-dataset/data) and includes the following columns:

| Column | Description |
|--------|-------------|
| `title` | The article title |
| `content` | Full text of the article |
| `category` | Article category (optional, for evaluation purposes) |

### Sample Data

| Title | Content | Category |
|-------|---------|----------|
| Article 1 | Lorem ipsum dolor sit amet... | World |
| Article 2 | Consectetur adipiscing elit... | Sports |

---

## Methodology

### Data Preprocessing

- Fill missing values in `content` with empty strings
- Convert all text to lowercase for consistency
- Handle missing `title` and `category` fields

### TF-IDF Vectorization

- Convert article content into numerical vectors using **Term Frequency-Inverse Document Frequency (TF-IDF)**
- Limit features to **5,000** for computational efficiency

### Cosine Similarity

- Compute pairwise **cosine similarity** between all articles
- Captures textual closeness between articles based on content overlap
- Ranges from 0 (completely dissimilar) to 1 (identical)

### K-Means Clustering

- Articles are grouped into **5 clusters** using K-Means algorithm
- Cluster similarity metric:
  - **1** if two articles are in the same cluster
  - **0** if articles are in different clusters
- Enhances recommendations by grouping semantically related articles

### Hybrid Recommendation Model

The hybrid model combines cosine similarity and cluster similarity using weighted averaging:

$$\text{Hybrid Similarity} = \alpha \cdot \text{Cosine Similarity} + \beta \cdot \text{Cluster Similarity}$$

Where: $\alpha + \beta = 1$

**Default weights:**
- $\alpha = 0.6$ (cosine similarity weight)
- $\beta = 0.4$ (cluster similarity weight)

The **top 5 articles** with the highest hybrid similarity scores are recommended to the user.

---

## Model Evaluation

The model's performance is evaluated using Precision@5, Recall@5, and F1-Score metrics across different similarity approaches:

| Method | Precision@5 | Recall@5 | F1 Score |
|--------|-------------|----------|----------|
| TF-IDF + Cosine | 0.362286 | 0.018297 | 0.034835 |
| TF-IDF + Euclidean | 0.362286 | 0.018297 | 0.034835 |
| **Hybrid (Cosine + Cluster)** | **0.379143** | **0.019149** | **0.036456** |

The hybrid model outperforms traditional methods by leveraging clustering information to improve recommendation relevance.

---

## Web Application

The application is built using **Streamlit**, providing an intuitive interface for exploring article recommendations.

### Features

- 📄 **Select an article** to get personalized recommendations
- ⚖️ **Adjust weights dynamically** for cosine (α) and cluster (β) similarity
- 🏷️ **View cluster ID** of the selected article
- 📋 **Display top 5 recommended articles** with similarity scores
- ☁️ **Generate word cloud** visualization of recommended content
- 📊 **Optional evaluation metrics** display

### Live Demo

🔗 [Try the Hybrid Article Recommendation System](https://cluster-enhanced-article-recommendation-system-lshvdvd8zswibui.streamlit.app/)

You can access and interact with the deployed application without running it locally.

---

## Installation & Usage

### Prerequisites
Ensure you have **Python 3.8+** and **pip** installed on your system.

### Clone the Repository
```bash
git clone https://github.com/Punam-rmstu/Cluster-Enhanced-Article-Recommendation-System.git
cd Cluster-Enhanced-Article-Recommendation-System
```

### Install Dependencies
Install all required packages from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Run the Streamlit Web Application
Launch the application with the following command:
```bash
streamlit run app.py
```

### Access the Application
Once the command executes, the application will automatically open in your default browser. If not, manually navigate to:
```
http://localhost:8501
```

### Optional: Deploy on Streamlit Cloud
- Push your repository to GitHub
- Connect your GitHub account to [Streamlit Cloud](https://streamlit.io/cloud)
- Deploy directly from your repository with a single click

### Optional: Deploy on HuggingFace Spaces
- Create a new Space on HuggingFace
- Connect your GitHub repository
- Configure the deployment settings and launch

---

## Future Enhancements

- **User-Based Collaborative Filtering:** Integrate collaborative filtering to combine content-based and user preference-based recommendations for improved personalization.
- **Topic Modeling with LDA:** Implement Latent Dirichlet Allocation (LDA) for deeper topic analysis and enhanced relevance in recommendations.
- **Real-Time Dataset Updates:** Support API integration for dynamic dataset updates without manual intervention.
- **Advanced Visualizations:** Implement interactive plots using Plotly or Dash for better data exploration and insights.
- **Multi-Language Support:** Extend the system to process articles in multiple languages.
- **Performance Optimization:** Optimize similarity calculations using approximate nearest neighbors (ANN) for large-scale datasets.
- **User Feedback Loop:** Incorporate user feedback mechanisms to continuously improve recommendations over time.

---

## License

**© 2026 Punam Kanungoe. All Rights Reserved.**

This project is provided as-is for educational and reference purposes.
You may **view and use** the code, but **modifications are strictly prohibited** without written permission.
