import streamlit as st
from data import EntData
from model import Model


DATA_DIR = "./../emerging_entities_17"
train_file = f"{DATA_DIR}/wnut17train.conll"
dev_file = f"{DATA_DIR}/emerging.dev.conll"
test_file = f"{DATA_DIR}/emerging.test.annotated"

if 'model' not in st.session_state:
    st.session_state['model'] = ''
#if 'sent' not in st.session_state:
#    st.session_state['sent'] = ''


lstm_model_path = "./../models/221_1156/4"
roberta_model_path = "./../models/317_1758/6"
roberta_bas_model_path = "./../models/410_136/6"

st.header("Emerging Entities Tagging: A 6390 Project")
model_option = st.selectbox("Which model needs to be tested?", ('LSTM','RoBERTa','RoBERTa-basilisk'))

if st.session_state['model']!=model_option:
    if model_option == "LSTM":
        data = EntData(train_file,dev_file, test_file)
        model = Model("lstm_char",data)
        model.load_model(lstm_model_path)
    elif model_option == "RoBERTa":
        data = EntData(train_file,dev_file, test_file)
        model = Model("roberta", data)
        model.load_model(roberta_model_path)
    elif model_option == "RoBERTa-basilisk":
        data = EntData("basilisk_train2.txt",dev_file, test_file)
        model = Model("roberta",data)
        model.load_model(roberta_bas_model_path)
    st.session_state['model'] = model_option
    st.session_state['model_val'] = model


sent = st.text_input('Input Sentence',"The paper is with Terry .")

st.write()
st.write("Model Prediction:")
#pred_tags = model.predict(sent.split())
pred_tags = st.session_state['model_val'].predict(sent.split())
c1, c2 = st.columns(2)
c1.markdown("**Word**")
c2.markdown("**Predicted Tag**")
for ix, word in enumerate(sent.split()):
    c1.write(word)
    c2.write(f"{pred_tags[ix]}")

