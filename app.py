import pandas as pd
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = pd.read_csv("clean.csv")

# Title and description
st.title("Amazon Recommender System")
st.write("This system uses content-based similarity to recommend products from Amazon.")

# Show list of products button
if st.button('List of Products'):
    st.write(data[["Product", "Price", "Rating"]])

# Get user input
st.subheader("What would you like to purchase?")
product_name = st.text_input("Enter the product name:")

# Recommender function
def recommender(product_name, data):
    # Check if the product exists in the dataset
    if product_name not in data["Product"].values:
        st.write("Product not found. Please enter a valid product name.")
        return

    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data["About"].fillna(""))

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get product index
    product_index = data[data["Product"] == product_name].index[0]

    # Calculate similarity scores
    similarity_scores = pd.DataFrame(cosine_sim[product_index], columns=["score"])
    product_indices = similarity_scores.sort_values("score", ascending=False).index[1:6]

    # Return recommended products
    recommended_products = data.loc[product_indices, ["Product", "Price", "Rating"]]
    return recommended_products

# Display recommended products
if product_name:
    recommended = recommender(product_name, data)
    if recommended is not None:
        st.write("You may also like:")
        st.table(recommended)

