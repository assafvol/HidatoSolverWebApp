import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from hidato_reader import read_and_solve_hidato

st.set_option("deprecation.showfileUploaderEncoding", False)
@st.cache(allow_output_mutation=True)
def load_model():
    m = tf.keras.models.load_model('my_model2')
    print('model loaded')  # just to keep track in your server
    return m


model = load_model()

st.write("""
        # Hidato Solver
        """
         )

file = st.file_uploader("Please upload an image of a Hidato", type=['jpg', 'png', 'jpeg'])
if file is not None:
    img = Image.open(file)
    img = img.convert('RGB')
    st.write("The hidato uploaded is:")
    angle = st.slider("choose rotation angle (rotate the hidato so that it is oriented vertically)", min_value=0, max_value=360, step=1)
    rot_img = img.rotate(-angle)
    st.image(rot_img)
    selection = st.selectbox(label="How do you want the solved hidato to be displayed?",
                             options=['Overlaid on the original image','In a new image'])
    on_orig = True if selection == 'Overlaid on the original image' else False
    if st.button("Solve hidato"):
        img_arr = np.array(rot_img)
        solved_img = read_and_solve_hidato(img_arr, model=model, on_original_image=on_orig)
        st.write("The solved hidato is:")
        st.image(solved_img)