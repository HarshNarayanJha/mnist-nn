import keras
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Prediction using DL")

cnn = st.sidebar.toggle("Use CNN Model", value=True, help="Toggle to use CNN trained model")


@st.cache_resource
def load_model(cnn) -> keras.Model:
    if cnn:
        model = keras.models.load_model("./mnist_digit_cnn.keras")
    else:
        model = keras.models.load_model("./mnist_digit.keras")
    assert isinstance(model, keras.Model)
    return model


def predict(data):
    model = load_model(cnn)
    pred = model.predict(data)
    return pred


st.title("MNIST Digit Detection using Deep Learning Model")
st.markdown("### Draw any digit, we will recognize it")
st.info("Use the canvas below to draw the digit")

st.sidebar.info("Loaded the CNN Model" if cnn else "Loaded the NN Model")

canvas_result = st_canvas(stroke_color="white", stroke_width=8, width=192, height=192)

if canvas_result.image_data is not None and canvas_result.image_data.any():
    with st.spinner(text="Thinking"):
        img = Image.fromarray(canvas_result.image_data)
        img = img.resize((28, 28))
        img = img.convert("L")
        imgs = np.array([np.array(img)])
        img.close()

        pred = predict(imgs)
        preds = pred.argmax(axis=-1)
        st.success(f"It is **{preds[0]}**", icon="✨")
        st.markdown(f"## {preds[0]}")
