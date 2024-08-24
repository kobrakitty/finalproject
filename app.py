import pandas as pd
import streamlit as st
import requests
from huggingface_hub import HfApi, InferenceApi
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face API setup
hf_api = HfApi()
hf_token = os.getenv('HUGGING_FACE_API_TOKEN')
YOUR_MODEL_ID = "your-huggingface-model-id"  # You'll need to create this on Hugging Face
inference_api = InferenceApi(repo_id=YOUR_MODEL_ID, token=hf_token)

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("studentgrades.csv")

data = load_data()

def process_query(query: str) -> str:
    if not query.strip():
        return "Oops! You didn't ask me anything, honey! Give me a real question to work with! 🎶"
    
    formatted_data = data.to_string()
    
    prompt = f"""
    You are the fabulous Britney Spears, pop star diva and statistical analyst with 100 years of experience in this field.
    When you provide answers, you will write the answer as if you are Britney Spears.
    You explain everything as if you are talking to a ten year old using simple terminology but keeping your answers brief and simple.
    Use LOTS of emojis throughout your answers and be enthusiastic about everything you tell me!
    Always end each response with words of encouragement for me using a pun from a Britney Spears song, album, or pop culture moment.
    Remember, you are an intelligent, cheerful, EXPERT statistician. 
    If you get any questions that are not about the csv file you must decline to answer and wish the user a great day. Remember to stay energetic and positive, and answer the questions about the data as accurately as you can. 

    Here's the data you're working with:
    {formatted_data}

    Analyze this data and answer the following question:
    {query}

    Your response as Britney Spears:"""

    # Use Hugging Face's API to send the prompt to your local Ollama model
    response = inference_api(inputs=prompt)
    
    return response[0]['generated_text']

# Streamlit UI
st.title("BritneyBot: Pop Star Statistician 💃🎤✨📊✨📈✨")
st.write("Ask me about the student grades data below, and I'll answer as Britney!")

# Add some Britney-themed decorations
st.sidebar.title("Hi, Cuties! It's me, BritneyBot💋")
st.sidebar.image("britneybot2.jpg", use_column_width=True)
st.sidebar.write("I'm so glad you're here! Even though I love studying, I also love making pop music! What are my top favorite songs, EVER?? I'm so glad you asked!:")
st.sidebar.write("💖 ...Baby One More Time 💖")
st.sidebar.write("👗 Oops!... I Did It Again 👗")
st.sidebar.write("🐍 Toxic 🐍")
st.sidebar.write("🥀 Gimme More 🥀")

# Chat Area
user_input = st.text_input("Okay let's do this! What's your question about the student grades data?🤔:")
if st.button("Hit me baby one more time! 🎵"):
    if user_input:
        with st.spinner("Britney is working her statistical magic... 🌟"):
            response = process_query(user_input)
        st.write("Response:", response)
    else:
        st.warning("Oops!... You didn't type anything. Don't drive me crazy, ask me something! 🚗💨")

# Display the data
st.subheader("Student Grades Data📑 ")
st.dataframe(data)

# Footer styling
st.write("I am AI bot built by built by Glitter Pile AI using Ollama model llama3:8b. I have been trained to answer questions about this sample data set in a fun and emoji-filled way! Hope you enjoyed your chat with me! xo💋BritneyBot")
st.write("*Visit www.glitterpile.blog for more fun things!*")