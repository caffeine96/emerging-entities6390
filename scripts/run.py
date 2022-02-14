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

    model = Model(exp_name,data)
    model.train()
