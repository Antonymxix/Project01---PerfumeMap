# Fragrance Space Visualization
An AI-driven approach to mapping olfactory relationships between designer perfumes.

## Project Overview
This project visualizes the semantic similarity of 100 designer fragrances. Instead of relying on manual categorization, it utilizes Natural Language Processing (NLP) to analyze fragrance notes and accords, mapping them into a two-dimensional vector space.

## Technical Implementation
* **Embeddings:** Used the `all-MiniLM-L6-v2` transformer model to convert textual fragrance descriptions into 384-dimensional dense vectors.
* **Dimensionality Reduction:** Implemented Principal Component Analysis (PCA) to project high-dimensional embeddings into 2D coordinates for visualization.
* **Clustering:** Fragrances are automatically grouped based on their olfactory profiles to reveal hidden patterns in scent families.
* **Frontend:** Developed an interactive dashboard using Streamlit and Plotly for real-time data exploration.

## Core Features
* Interactive scatter plot mapping similar scents in close proximity.
* Semantic search and analysis of fragrance notes.
* Modern Glassmorphism UI design.

## Setup and Usage
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the application:
   `streamlit run app.py`
