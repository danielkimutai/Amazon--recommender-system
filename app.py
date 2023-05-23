import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np 
import regex
import re
import string
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
st.title("Amazon Recommender System")
st.write("This is a system that uses content based similarity to recommend products.The data mainly contains Amazon Products")
if st.button('Show dataframe'):
    data = pd.read_csv("clean.csv")        
    data[:10]
st.header("Most Expensive Products")
data = pd.read_csv("clean.csv")  
df_group10 = data.groupby("subcategory")["Price"].mean().sort_values(ascending = False)[:10]
# Reset the index
df_group10 = df_group10.reset_index()
st.bar_chart(df_group10,x="subcategory",y="Price")
def wordcloud(data,column):
    text = ' '.join(data[column])
    wordcloud = WordCloud(width = 800, height = 800,
                    min_font_size = 10).generate(text)
    # plot the WordCloud image                      
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.title(f'{column} word cloud')
    st.pyplot()
wordcloud(data,"Product")
        