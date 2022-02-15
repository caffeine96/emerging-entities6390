import pandas as pd
import torch
from tqdm import tqdm

from bilstmcrf import BiLSTMCRF

GPU_ID = '1'
torch.manual_seed(42)
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

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


    
    
    def train(self, max_ep=10, lr=0.001):
        # Tokenize and convert words into sub-words/characters 
        # and sub-words/characters to indices
        self.data.train_data['inp_tok'], self.data.train_data['inp_ind'] = \
                self.model.tokenize(self.data.train_data['inp'])
        self.data.dev_data['inp_tok'], self.data.test_data['inp_ind'] = \
                self.model.tokenize(self.data.dev_data['inp'])
        self.data.test_data['inp_tok'], self.data.test_data['inp_ind'] = \
                self.model.tokenize(self.data.test_data['inp'])

        train_set   = pd.DataFrame.from_dict(self.data.train_data)
        dev_set     = pd.DataFrame.from_dict(self.data.dev_data)
        test_set    = pd.DataFrame.from_dict(self.data.test_data)
       
        optimizer = torch.optim.Adam(list(self.model.parameters()),lr=lr)

        for ep in range(1,max_ep+1):
            print(f"Epoch {ep}\n")
            ep_tr_loss = 0
            for train_ix, row in tqdm(train_set.iterrows()): 
                optimizer.zero_grad()
                loss = self.model.compute_loss(row)
                loss.backward()
                optimizer.step()
                ep_tr_loss += loss.item()

            print(f"Average Train Loss: {ep_tr_loss/len(train_set)}\n")

        
