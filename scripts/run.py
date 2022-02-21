import argparse
import pandas as pd
from data import EntData
from model import Model

DATA_DIR = "./../emerging_entities_17"

def parse_args(parser):
    parser.add_argument('--train_file',default=f"{DATA_DIR}/wnut17train.conll", type =str)
    parser.add_argument('--dev_file',default=f"{DATA_DIR}/emerging.dev.conll", type =str)   
    parser.add_argument('--test_file',default=f"{DATA_DIR}/emerging.test.annotated", type =str)
    # Experiment Name suggests the experimental configuration
    # Below are the valid values
    # 1. "lstm_char" - LSTM-CRF model with character embeddings (Phase 1) 
    parser.add_argument('--exp_name', default="lstm_char", type =str)
    parser.add_argument('--run_type', default="predict", type=str)
    parser.add_argument('--eval_split', default="dev", type=str)
    parser.add_argument('--pred_sentence', default="The paper was with Terry .", type=str)
    parser.add_argument('--model_path', default="./../models/test/9", type=str)

    return parser



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    args = vars(parser.parse_args())

    data = EntData(args['train_file'],args['dev_file'],args['test_file'])
    
    pred_sentence = args['pred_sentence']

    model = Model(args['exp_name'], data)
    if args['run_type'] == "train":
        model.train()
    else:
        model_path = args['model_path']
        model.load_model(model_path)
        if args['run_type'] == "eval":
            if args['eval_split'] == "dev":
                model.data.dev_data['inp_tok'], model.data.dev_data['inp_ind'] =  model.model.tokenize(model.data.dev_data['inp'])
                split_data = pd.DataFrame.from_dict(model.data.dev_data)
            elif args['eval_split'] == "train":
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
