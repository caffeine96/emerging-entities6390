import pandas as pd
from data import EntData
from model import Model

if __name__ == "__main__":
    DATA_DIR = "./../emerging_entities_17"
    train_file = f"{DATA_DIR}/wnut17train.conll"
    dev_file = f"{DATA_DIR}/emerging.dev.conll"
    test_file = f"{DATA_DIR}/emerging.test.annotated"

    data = EntData(train_file,dev_file,test_file)
    
    # Experiment Name suggests the experimental configuration
    # Below are the valid values
    # 1. "lstm_char" - LSTM-CRF model with character embeddings (Phase 1) 
    exp_name = "lstm_char"

    run_type = "predict"
    eval_split = "dev"
    pred_sentence = "The paper was with Terry ."
    pred_sentence = "Succession is a great show ."

    model = Model(exp_name, data)
    if run_type == "train":
        model.train()
    else:
        model_path = "./../models/test/9"
        model.load_model(model_path)
        if run_type == "eval":
            if eval_split == "dev":
                model.data.dev_data['inp_tok'], model.data.dev_data['inp_ind'] =  model.model.tokenize(model.data.dev_data['inp'])
                split_data = pd.DataFrame.from_dict(model.data.dev_data)
            elif eval_split == "train":
                model.data.train_data['inp_tok'], model.data.train_data['inp_ind'] =  model.model.tokenize(model.data.train_data['inp'])
                split_data = pd.DataFrame.from_dict(model.data.train_data)
            else:
                model.data.test_data['inp_tok'], model.data.test_data['inp_ind'] =  model.model.tokenize(model.data.test_data['inp'])
                split_data = pd.DataFrame.from_dict(model.data.test_data)
        
            metrics = model.evaluate(split_data, save=False)
            print(metrics)
        else:
            pred_tags = model.predict(pred_sentence.split())
            for ix, word in enumerate(pred_sentence.split()):
                print(f"{word}\t{pred_tags[ix]}")
