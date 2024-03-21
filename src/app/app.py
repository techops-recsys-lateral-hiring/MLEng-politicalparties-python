import streamlit as st
import requests

def get_prediction(abstract):
    # Assuming your endpoint returns a JSON with a field named 'prediction'
    response = requests.post("http://model_inference_endpoint:8000/get-prediction/", json={"input_texts": abstract})
    if response.status_code == 200:
        # Parse the JSON response
        return response.text
    else:
        return "Error: Could not get prediction."

# Streamlit page configuration
st.set_page_config(page_title="Tweet Classifier", layout="wide")

# Streamlit UI components
st.title("Classify your tweet")

# User inputs the tweet
tweet_input = st.text_input("Enter your tweet", "")

# Button to trigger prediction
if st.button("Classify Tweet"):
    # Get prediction
    prediction = get_prediction(tweet_input)
    
    # Display the prediction
    st.write("Prediction:", prediction)

