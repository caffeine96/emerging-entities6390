import streamlit as st
from data import EntData
from model import Model


DATA_DIR = "./../emerging_entities_17"
train_file = f"{DATA_DIR}/wnut17train.conll"
dev_file = f"{DATA_DIR}/emerging.dev.conll"
test_file = f"{DATA_DIR}/emerging.test.annotated"

data = EntData(train_file,dev_file, test_file)


model = Model("lstm_char",data)
model_path = "./../models/221_1156/4"
model.load_model(model_path)

st.header("Emerging Entities Tagging: A 6390 Project")
sent = st.text_input('Input Sentence',"The paper is with Terry .")

st.write()
st.write("Model Prediction:")
pred_tags = model.predict(sent.split())
c1, c2 = st.columns(2)
c1.markdown("**Word**")
c2.markdown("**Predicted Tag**")
for ix, word in enumerate(sent.split()):
    c1.write(word)
    c2.write(f"{pred_tags[ix]}")

