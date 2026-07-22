import streamlit as st
from pathlib import Path
script_location = Path(__file__).absolute().parent.parent

st.write("# Predicting customer reorder time")

st.write("### Overview")
st.write ("This project predicts the timeframe in which a customer is likely to reorder based on their previous shopping experience.The model can be found by clicking 'Prediction' on the left. ")


col1, col2 = st.columns(2)
col1.metric(label="**Accuracy**", value = "55.8%")
col2.metric(label="**Macro F1**", value = "0.44")

st.write ("### Top features")
st.image(script_location / 'Figures/feature_importance_graph.png')

st.write ("### Confusion Matrix")
st.image(script_location /'Figures/confusion_matrix_graph.png')

