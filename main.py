import streamlit as st
import tensorflow as tf
import numpy as np

# --------- Page Config ----------
st.set_page_config(
    page_title="Fruits & Vegetables Recognition",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------- Custom CSS ----------
st.markdown("""
    <style>
    .main {
        background-color: #f9fafc;
    }
    .stButton>button {
        border-radius: 12px;
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        padding: 0.6em 1.2em;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# --------- TensorFlow Model Prediction ----------
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element


# --------- Sidebar ----------
st.sidebar.title("🍏 Dashboard")
app_mode = st.sidebar.radio("Navigate", ["🏠 Home", "📖 About Project", "🔮 Prediction"])


# --------- Main Pages ----------
# Home Page
if app_mode == "🏠 Home":
    col1, col2 = st.columns([1,2])
    with col1:
        st.image("home_img.jpg", use_column_width=True)
    with col2:
        st.markdown("<h2 style='color:#2e7d32;'>FRUITS & VEGETABLES RECOGNITION SYSTEM</h2>", unsafe_allow_html=True)
        st.write("Upload an image of a fruit or vegetable and let our AI predict what it is 🍎🥦🥕.")


# About Project Page
elif app_mode == "📖 About Project":
    st.markdown("<h2 style='color:#2e7d32;'>📂 About Project</h2>", unsafe_allow_html=True)
    st.subheader("📊 About Dataset")
    st.text("This dataset contains images of the following food items:")

    st.code("fruits: banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables: cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")

    st.subheader("📂 Dataset Content")
    st.text("1️⃣ train (100 images each)\n2️⃣ test (10 images each)\n3️⃣ validation (10 images each)")


# Prediction Page
elif app_mode == "🔮 Prediction":
    st.markdown("<h2 style='color:#2e7d32;'>🔮 Model Prediction</h2>", unsafe_allow_html=True)
    test_image = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

    if test_image:
        st.image(test_image, width=250, caption="Uploaded Image", use_column_width=False)

        if st.button("✨ Predict"):
            with st.spinner("Analyzing... ⏳"):
                result_index = model_prediction(test_image)
                # Reading Labels
                with open("labels.txt") as f:
                    content = f.readlines()
                labels = [i.strip() for i in content]

            st.markdown(
    f"""
    <div class='prediction-box'>
        <h3 style="color:#2e7d32;">✅ Model Prediction:</h3>
        <h3><span style='color:#d32f2f; font-weight:700;'>{labels[result_index]}</span></h3>
    </div>
    """,
    unsafe_allow_html=True
)
