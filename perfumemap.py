import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.express as px

# 1. configurate the page
st.set_page_config(page_title="Perfume Map", layout="wide")

# 2. loading the data
@st.cache_resource # so the model doesnt reload after every click
def load_data():
    df = pd.read_csv('perfumes.csv')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # convert text in numbers (Embeddings) 
    descriptions = df['Brand'] + " " + df['Main_Notes'] + " " + df['Accords']
    embeddings = model.encode(descriptions.tolist())
    
    # convert the 384 numbers to x and y coordinates
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]
    
    return df

df = load_data()

# 3. UI
st.title("Designer Fragrance Map")

# 4. map display
fig = px.scatter(df, x='x', y='y', 
                 text='Name', 
                 color='Accords',
                 hover_data=['Brand', 'Main_Notes'])

st.plotly_chart(fig, use_container_width=True)