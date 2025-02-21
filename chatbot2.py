import streamlit as st
import pandas as pd
import numpy as np
import serial
import time
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup the port for the Arduino
ser = None
try:
    ser = serial.Serial('/dev/ttyUSB0', 9600)  # Change '/dev/ttyUSB0' to your Arduino port
except serial.SerialException:
    st.warning("Arduino not connected. You can enter values manually.")

# Training data for soil quality
data = {
    "pH": [5.5, 6.0, 7.0, 8.0, 6.5, 7.5, 4.8, 6.2, 7.3, 5.9, 6.8, 5.3, 6.7, 5.1, 7.1],
    "Humidity": [35, 42, 55, 65, 47, 60, 20, 30, 70, 50, 55, 40, 52, 33, 58],
    "Conductivity": [0.3, 0.7, 1.2, 1.5, 0.9, 1.3, 0.2, 0.5, 1.0, 0.8, 1.4, 0.4, 1.1, 0.6, 0.9],
    "Quality": ["Poor", "Medium", "Good", "Poor", "Good", "Good", "Poor", "Medium", "Good", "Medium", "Good", "Poor", "Medium", "Poor", "Good"]
}

df = pd.DataFrame(data)
quality_mapping = {"Poor": 0, "Medium": 1, "Good": 2}
df["Quality"] = df["Quality"].map(quality_mapping)

# Train the classification model
X = df[["pH", "Humidity", "Conductivity"]]
y = df["Quality"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to analyze soil quality
def analyze_soil(pH, humidity, conductivity):
    # Check if the parameters are within optimal ranges
    if 5.5 <= pH <= 7.5 and 40 <= humidity <= 70 and 0.5 <= conductivity <= 1.5:
        quality_text = "Good"
        status_time = "It is a good time to plant!"
        improvement_advice = []
    else:
        predicted_quality = model.predict([[pH, humidity, conductivity]])[0]
        quality_text = {0: "Poor", 1: "Medium", 2: "Good"}[predicted_quality]

        reasons_not_good_time = []
        improvement_advice = []

        if not (5.5 <= pH <= 7.5):
            reasons_not_good_time.append(f"pH is at {pH}, which is {'too low' if pH < 5.5 else 'too high'}. It should be between 5.5 and 7.5.")
            if pH < 5.5:
                improvement_advice.append("Add lime or wood ash to raise pH.")
            else:
                improvement_advice.append("Add aluminum sulfate or sulfur to lower pH.")

        if not (40 <= humidity <= 70):
            reasons_not_good_time.append(f"Humidity is at {humidity}%, which is {'too low' if humidity < 40 else 'too high'}.")
            if humidity < 40:
                improvement_advice.append("Increase irrigation or apply mulching to conserve moisture.")
            else:
                improvement_advice.append("Reduce irrigation and improve soil drainage.")

        if not (0.5 <= conductivity <= 1.5):
            reasons_not_good_time.append(f"Conductivity is at {conductivity} dS/m, which is {'too low' if conductivity < 0.5 else 'too high'}.")
            if conductivity < 0.5:
                improvement_advice.append("Add fertilizers that increase salinity in a controlled manner.")
            else:
                improvement_advice.append("Wash the soil with water to reduce salinity.")

        good_time = len(reasons_not_good_time) == 0
        status_time = "It is a good time to plant!" if good_time else "It is not a good time to plant. Reasons: " + "; ".join(reasons_not_good_time)

    return quality_text, status_time, improvement_advice

# Load the chatbot model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chatbot_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Function to get the chatbot's response
def get_chatbot_response(user_input, chat_history):
    input_text = chat_history + user_input + tokenizer.eos_token
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    response_ids = chatbot_model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Streamlit interface
st.title("💬 AgriBot Soil Quality 🌱")

# Initialize cached values for sliders
if 'pH' not in st.session_state:
    st.session_state.pH = 6.5
if 'humidity' not in st.session_state:
    st.session_state.humidity = 50
if 'conductivity' not in st.session_state:
    st.session_state.conductivity = 0.8
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ""

# Sliders for soil parameters
st.subheader("📊 Analyze Soil Quality")
st.session_state.pH = st.slider("pH Level", min_value=3.0, max_value=9.0, step=0.1, value=st.session_state.pH)
st.session_state.humidity = st.slider("Humidity (%)", min_value=10, max_value=80, step=1, value=st.session_state.humidity)
st.session_state.conductivity = st.slider("Conductivity (dS/m)", min_value=0.1, max_value=2.0, step=0.1, value=st.session_state.conductivity)

if st.button("🔍 Analyze"):
    quality, time_status, advice = analyze_soil(st.session_state.pH, st.session_state.humidity, st.session_state.conductivity)
    st.success(f"📊 **Soil Quality:** {quality}")
    st.info(f"💡 **Evaluation:** {time_status}")
    if advice:
        st.info(f"💡 **Advice for Improving Parameters:** {', '.join(advice)}")
    else:
        st.info("💡 **No additional actions needed to improve parameters.**")

# Chatbot section
st.subheader("💬 Talk to AgriBot ")
user_input = st.text_input("Do you have questions about soil analysis?")
if st.button("Send"):
    st.session_state.chat_history += user_input + tokenizer.eos_token
    bot_response = get_chatbot_response(user_input, st.session_state.chat_history)
    st.session_state.chat_history += bot_response + tokenizer.eos_token
    st.text_area("Chatbot Response:", value=bot_response, height=150)

# Countdown timer
st.subheader("⏳ Data Update")
counter = st.empty()
for i in range(60, 0, -1):
    counter.text(f"Updating in {i} seconds...")
    time.sleep(1)

# Receive data from Arduino or allow manual entry if not connected
if ser is not None and ser.is_open:
    line = ser.readline().decode('utf-8').strip()  # Read data from Arduino
    pH, humidity, conductivity = map(float, line.split(','))
    st.session_state.pH = pH
    st.session_state.humidity = humidity
    st.session_state.conductivity = conductivity
else:
    st.warning("Arduino not connected. You can enter values manually.")
