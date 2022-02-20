import numpy as np
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm

from bilstmcrf import BiLSTMCRF

GPU_ID = '1'
torch.manual_seed(42)
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "./../models/"

class Model():
    def __init__(self, exp_name, data):
        """ Constructor for the model class
        Inputs
        -------------
        exp_name    - str. Name of the experiment
        data        - EntData. Instance of with processed inputs
        """
        self.data = data
        self.exp_name = exp_name

        if self.exp_name == "lstm_char":
            self.model = BiLSTMCRF(data,device=device).to(device)
            self.model.prep_tokenizer(data)


    
    
    def train(self, max_ep=100, lr=0.001, save_model=True):
        # Tokenize and convert words into sub-words/characters 
        # and sub-words/characters to indices
        folder_name = "test/"
        if save_model:
            Path.mkdir(Path(MODEL_DIR+folder_name), exist_ok=True, parents=True)
        self.data.train_data['inp_tok'], self.data.train_data['inp_ind'] = \
                self.model.tokenize(self.data.train_data['inp'])
        self.data.dev_data['inp_tok'], self.data.dev_data['inp_ind'] = \
                self.model.tokenize(self.data.dev_data['inp'])
        self.data.test_data['inp_tok'], self.data.test_data['inp_ind'] = \
                self.model.tokenize(self.data.test_data['inp'])

        train_set   = pd.DataFrame.from_dict(self.data.train_data)
        dev_set     = pd.DataFrame.from_dict(self.data.dev_data)
        test_set    = pd.DataFrame.from_dict(self.data.test_data)
       
        optimizer = torch.optim.Adam(list(self.model.parameters()),lr=lr)

        save = True
        print_log = False
        best_metric = 0
        for ep in range(1,max_ep+1):
            print(f"Epoch {ep}\n")
            ep_tr_loss = []
            for train_ix, row in tqdm(train_set.iterrows()): 
                optimizer.zero_grad()
                #if train_ix == 2:
                #    print_log = True
                loss = self.model.compute_loss(row, print_log)
                loss.backward()
                optimizer.step()
                ep_tr_loss.append(loss.item())
                
                #if train_ix == 100:
                #    print(ep_tr_loss)
                #    print(row['inp_tok'])
                #    print(np.mean(ep_tr_loss))
                #    exit()
                #   break

            print(f"Average Train Loss: {np.mean(ep_tr_loss)}\n")
            dev_metrics = self.evaluate(dev_set, save, best_metric)

            if dev_metrics['total']['f1'] > best_metric:
                best_metric = dev_metrics['total']['f1']
            
                torch.save({
                    'epoch' : ep,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'metrics': dev_metrics
                    }, f"{MODEL_DIR}{folder_name}{ep}")


    def evaluate(self, df, save=False, best_metric=0):
        all_pred = []
        all_gold = []
        all_word = []
        for ix, row in tqdm(df.iterrows()):
            preds ,out = self.model.predict(row)
            all_pred.extend(preds[0].tolist())
            all_gold.extend(row['label'])
            all_word.extend(row['inp'])
        
        eval_metrics = self.calc_f1(all_gold,all_pred)

        print(f"""Dev Metric: {eval_metrics['total']['f1']}\n""")

        if save and (eval_metrics['total']['f1']>best_metric):
            with open("dev_preds.txt","w+") as f:
                for ix in range(len(all_gold)):
                    g_lab = self.data.labels[all_gold[ix]]
                    p_lab = self.data.labels[all_pred[ix]]
                    line = f"{all_word[ix]}\t{g_lab}\t{p_lab}\n"
                    f.write(line)
        return eval_metrics




    def load_model(self,model_path):
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
                





    def calc_f1(self, gold, preds):
        conv_gold = [self.data.labels[el] for el in gold]
        conv_pred = [self.data.labels[el] for el in preds]
        #print(self.data.labels)
    
        cf = {"total": np.zeros((2,2)),
                "corporation": np.zeros((2,2)),
                "person"  : np.zeros((2,2)),
                "creative-work"  : np.zeros((2,2)),
                "group": np.zeros((2,2)),
                "location": np.zeros((2,2)),
                "product": np.zeros((2,2))}
        
        for ix in range(len(conv_gold)):
            g_lab = conv_gold[ix]
            p_lab = conv_pred[ix]
            if (g_lab == p_lab) and g_lab!='O':
                key = "-".join(g_lab.split("-")[1:])
                cf[key][0][0] += 1
                cf["total"][0][0] += 1
            else:
                if g_lab!='O':
                    g_key = "-".join(g_lab.split("-")[1:])
                    cf[g_key][0][1] += 1
                    cf["total"][0][1] += 1
                
                if p_lab != 'O':
                    p_key = "-".join(p_lab.split("-")[1:])
                    cf[p_key][1][0] += 1
                    cf['total'][1][0] += 1

        metrics = {}
        for key in cf.keys():
            metrics[key] = {}
            tp = cf[key][0][0]
            fp = cf[key][1][0]
            fn = cf[key][0][1]

            if (tp+fp) == 0:
                pr = 0
            else:
                pr = tp/(tp+fp)
            metrics[key]['precision'] = pr

            if (tp+fn) == 0:
                re = 0
            else:
                re = tp/(tp+fn)
            metrics[key]['recall'] = re

            if (pr + re) == 0:
                f1 = 0
            else:
                f1 = (2*pr*re)/(pr+re)
            metrics[key]['f1'] = f1


        return metrics



    def predict(self,tokens):
        inp_dict = {}
        inp_dict['inp'] = tokens
        inp_tok, inp_ind = self.model.tokenize([tokens])
        inp_dict['inp_tok'] = inp_tok[0]
        inp_dict['inp_ind'] = inp_ind[0]
        
        preds, _ = self.model.predict(inp_dict)

        pred_final = preds[0].tolist()
        pred_final = [self.data.labels[p] for p in pred_final]

        return pred_final
