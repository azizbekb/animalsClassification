import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
plt=platform.system()
if plt=='Linux':pathlib.WindowsPath=pathlib.PosixPath

#title
st.title('Animal classification model')
st.markdown("bird, bear, fish classification model")

#post a picture
file=st.file_uploader('Upload image (bird, bear, fish)', type=['png', 'jpeg', 'gif', 'svg', 'webp', 'jfif'])

if file:
    st.image(file)

    #PIL convert
    img=PILImage.create(file)

    #model
    model=load_learner('animal_model.pkl')

    #prediction
    pred, pred_id, probs=model.predict(img)
    st.success(f"Predict: {pred}")
    st.info(f"Probability: {probs[pred_id]*100:.1f}%")

    #plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    fig.update_layout(
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
    st.plotly_chart(fig)
