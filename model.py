# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 23:26:06 2025

@author: sanket
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load saved models and data
with open('user_similarity_matrix.pkl', 'rb') as file:
    user_sim_matrix = pickle.load(file)

with open('user_item_matrix.pkl', 'rb') as file:
    user_item_matrix = pickle.load(file)

st.title('📚 Book Recommendation System')

# User input
user_id = st.number_input("Enter User ID", min_value=0, max_value=len(user_item_matrix)-1, step=1)

top_k = st.slider("Number of recommendations", 1, 10, 5)

if st.button('Get Recommendations'):
    if user_id in user_item_matrix.index:
        # Find the most similar user
        user_index = user_item_matrix.index.get_loc(user_id)
        similar_users = user_sim_matrix[user_index]
        most_similar_user = similar_users.argmax()

        # Recommend books the similar user has rated highly
        recommendations = user_item_matrix.columns[user_item_matrix.iloc[most_similar_user].argsort()[::-1][:top_k]].tolist()

        st.write(f"Top {top_k} recommendations for User {user_id}:")
        for idx, book in enumerate(recommendations, start=1):
            st.write(f"{idx}. {book}")
    else:
        st.error('User ID not found.')