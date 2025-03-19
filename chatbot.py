import streamlit as st
import json
import random
import pickle
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences

# Load intents
with open("intent.json") as file:
    data = json.load(file)

# Load trained model
model = keras.models.load_model("chat_model.keras")

# Load tokenizer object
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load label encoder object
with open("label_encoder.pickle", "rb") as enc:
    lbl_encoder = pickle.load(enc)

# Parameters
max_len = 20

# Streamlit UI
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("<h1 style='margin-bottom: 0px;'>SkinTwin</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0px;'>Your Virtual Skincare Twin 🫧🧴</h3>", unsafe_allow_html=True)

with col2:
    st.image("image.png", width=150)  # Adjust width as needed

    
st.write("Ask me anything about skincare!")

# Suggested queries
suggestions = [
    "How do I know my skin type?",
    "Best treatment for acne?",
    "How to remove dark spots?",
    "Best anti-aging skincare?",
    "What SPF should I use?",
    "How to build a skincare routine?",
    "Best skincare brands?",
    "Which moisturizer is best for dry skin?",
    "Nighttime skincare routine?"
]

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["text"])

# User input with suggestions
def get_user_input():
    selected_query = st.selectbox("Need a suggestion?", ["Type your own question..."] + suggestions)
    if selected_query == "Type your own question...":
        return st.chat_input("Your Question:")
    return selected_query

user_input = get_user_input()

if user_input:
    st.session_state["messages"].append({"role": "user", "text": user_input})
    
    # Process input
    result = model.predict(
        pad_sequences(tokenizer.texts_to_sequences([user_input]), truncating="post", maxlen=max_len)
    )
    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]
    
    # Find response
    response = "I didn't understand that."
    for i in data["intents"]:
        if i["tag"] == tag:
            response = random.choice(i["responses"])
            break
    
    # Store and display response
    st.session_state["messages"].append({"role": "assistant", "text": response})
    with st.chat_message("assistant"):
        st.write(response)