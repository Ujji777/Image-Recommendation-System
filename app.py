import streamlit as st
import requests
import os

API_URL = "http://127.0.0.1:8000/api/recommend/"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_images")

st.set_page_config(page_title="Artwork Recommender", layout="wide")
st.title(" Artwork Recommendation System")


if not os.path.exists(DATASET_DIR):
    st.error(f"Dataset folder not found: {DATASET_DIR}")
else:
    image_files = sorted(os.listdir(DATASET_DIR))

    st.subheader("Select artworks you like")

    selected_images = []
    cols = st.columns(8)  
    for idx, img_name in enumerate(image_files):
        with cols[idx % 8]:
            img_path = os.path.join(DATASET_DIR, img_name)
            st.image(img_path, width=120) 
            if st.checkbox("Like", key=img_name):
                selected_images.append(img_name)

    if st.button("Get Recommendations") and selected_images:
        liked_str = ",".join(selected_images)
        try:
            response = requests.get(API_URL, params={"liked": liked_str})
            if response.status_code == 200:
                recs = response.json().get("recommendations", [])

                if recs:
                    st.subheader("Recommended Artworks Based on User Preferences")
                    rec_cols = st.columns(8)
                    for i, rec in enumerate(recs):
                        with rec_cols[i % 8]:
                            rec_path = os.path.join(DATASET_DIR, rec)
                            if os.path.exists(rec_path):
                                st.image(rec_path, width=120)
                            st.caption(rec)
                else:
                    st.warning("No recommendations found.")
            else:
                st.error("Error fetching recommendations from API.")
        except Exception as e:
            st.error(f"Request failed: {e}")
