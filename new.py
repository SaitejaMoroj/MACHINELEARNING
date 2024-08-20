import streamlit as st
import pickle

# Load the model and data
model_path = 'laptop_price_prediction.pkl'
data_path = 'laptop_data.pkl'

try:
    with open(model_path, 'rb') as model_file:
        pipe = pickle.load(model_file)
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")

try:
    with open(data_path, 'rb') as data_file:
        df = pickle.load(data_file)
except FileNotFoundError:
    st.error(f"Data file not found: {data_path}")
    df = None  # Set df to None if data file is not found

# Check if df is loaded successfully
if df is not None:
    st.title('Laptop Price Prediction')

    # Brand
    company = st.selectbox('Brand', df['Brand'].unique())

    # Type
    type_ = st.selectbox('Type', df['typename'].unique())

    # RAM
    ram = st.selectbox('RAM in GB', [2, 4, 6, 8, 16, 32, 64])

    # Screen Size
    screen_size = st.number_input('Screen Size')

    # GPU Brand
    gpu_brand = st.selectbox('GPU Brand', df['GPU BRAND'].unique())

    # Condition
    new = st.selectbox('New', ['No', 'Yes'])
    open_box = st.selectbox('Open Box', ['No', 'Yes'])
    excellent_refurbished = st.selectbox('Excellent - Refurbished', ['No', 'Yes'])
    very_good_refurbished = st.selectbox('Very Good - Refurbished', ['No', 'Yes'])
    good_refurbished = st.selectbox('Good - Refurbished', ['No', 'Yes'])
    unknown = st.selectbox('Unknown', ['No', 'Yes'])
    used = st.selectbox('Used', ['No', 'Yes'])

    # Product
    product = st.selectbox('Product', df['product'].unique())

    # Screen Resolution
    resolution = st.selectbox('Screen Resolution', [
        '1280x720', '1366x768', '1920x1080', '2560x1440', '3840x2160', '7680x4320',
        '2880x1800', '2736x1824', '2732x2048', '2436x1125', '2688x1242', '2960x1440', '3200x1440'
    ])

    # When the 'Predict Price' button is clicked
    if st.button('Predict Price'):
        # Prepare input data for prediction
        input_data = [[company, type_, ram, screen_size, gpu_brand, new, open_box, excellent_refurbished,
                       very_good_refurbished, good_refurbished, unknown, used, product, resolution]]

        # You may need to preprocess the input_data before making a prediction, similar to how you did during training.
        # For simplicity, let's assume the preprocessing is already handled in the model pipeline.

        # Make prediction
        prediction = pipe.predict(input_data)

        # Display prediction
        st.write(f"The predicted price of the laptop is ${prediction[0]:.2f}")
else:
    st.error("Data not loaded. Please check the data file.")
