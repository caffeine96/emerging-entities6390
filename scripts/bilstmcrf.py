import torch
from tqdm import tqdm

torch.manual_seed(42)


class BiLSTMCRF(torch.nn.Module):
    def __init__(self, data, emb_size= 1024, hidden_size = 768, dropout=0.01, crf_layer=False,device="cpu"):
        """ Constructor for the BiLSTM CRF module
        Inputs
        --------------
        data - EntData. Instance with processed inputs
        emb_size - int. Embedding Dimensions
        hidden_size - int. Out-dimensions of the LSTM layer
        dropout     - float. Dropout at each layer
        """
        super(BiLSTMCRF, self).__init__()
        self.emb_size = emb_size
        self.crf_layer = crf_layer
        self.device = device
        # Intializing the LSTM layer. Inputs are embedded character
        # vectors which will be initailized once we know the vocabulary of 
        # characters.
        self.lstm = torch.nn.LSTM(emb_size, hidden_size,batch_first=True, num_layers=2)#, bidirectional=True)
        #self.lstm = torch.nn.Linear(emb_size, hidden_size*2)
        # This is the classification head on the LSTM layer
        self.fc1 = torch.nn.Linear(hidden_size,int(hidden_size/2))
        self.relu = torch.nn.ReLU() 
        self.fc2 = torch.nn.Linear(int(hidden_size/2),len(data.labels))

        self.softmax = torch.nn.Softmax(dim=-1)

        #self.label_sequence_validity = self.obtain_invalidity

    def prep_tokenizer(self,data):
        """ Computes the distinct number of characters in files.
        Initializes the embedding layer and sets the vocab size.
        """
        vocab = []
        for data_split in [data.train_data,data.dev_data,data.test_data]:
            for sent in data_split['inp']:
                for word in sent:
                    for charac in word:
                        if charac not in vocab:
                            vocab.append(charac)
        vocab.append("<unk>")
        self.vocab = vocab
        self.embedding_layer = torch.nn.Embedding(len(vocab),self.emb_size).to(self.device)
        
        if not self.crf_layer:
            count_list = [0]*len(data.labels)
            total = 0
            for sent_lab in data.train_data['label']:
                for lab in sent_lab:
                    total += 1
                    count_list[lab] += 1
            class_weights = torch.Tensor(list(map(lambda x : 1 - (x/total),count_list)))
            print(f"\nClass Weights: {class_weights}\n")
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(self.device)
    


    def tokenize(self, inp_data):
        """ Breaks down words into characters
        Inputs
        ---------------
        inp_data - List. List of tokenized sentences

        Outputs
        ----------------
        inp_tok - List. List of tokenized sentences
                    broken down character-wise 
        inp_ind - List. List of indices which maintains
                    information of which character belongs
                    to which word
        """
        inp_tok = []
        inp_ind = []
        
        for item in inp_data:
            sent_idx = []
            sent_pos = []
            ix = 0
            for i, word in enumerate(item):
                pos = []
                for charac in word:
                    try:
                        sent_idx.append(self.vocab.index(charac))
                    except ValueError:
                        sent_idx.append(len(self.vocab)-1)
                    pos.append(ix)
                    ix+=1
                sent_pos.append(pos)
            inp_tok.append(sent_idx)
            inp_ind.append(sent_pos)
        
        return inp_tok, inp_ind



    def _mean_pool(self, inp, ixs):
        mean_pooled_emb = None
        for ix_set in ixs:
            avg_pooled = torch.mean(inp[0,ix_set,:], dim=0, keepdim=True)
            if mean_pooled_emb == None:
                mean_pooled_emb = avg_pooled
            else:
                mean_pooled_emb = torch.cat((mean_pooled_emb,avg_pooled),dim=0)

        return mean_pooled_emb.unsqueeze(0)


    def forward_pass(self,inp, char_tokens_maps, print_log=False):
        emb_inp = self.embedding_layer(inp)
        lstm_out, _ = self.lstm(emb_inp)
        lstm_out_comb = self._mean_pool(lstm_out, char_tokens_maps)
        out = self.fc2(self.relu(self.fc1(lstm_out_comb)))

        return out 



    def compute_loss(self, data, print_log=False):
        """ Compute Loss for the training sample
        Inputs
        ---------------
        data- dict. Dictionary containing the training sample information
        """
        gold_lab = torch.Tensor(data['label']).to(torch.long).to(self.device)
        inp = torch.Tensor(data['inp_tok']).to(torch.long).unsqueeze(0).to(self.device)
        out = self.forward_pass(inp, data['inp_ind'], print_log)

        return self.loss_fn(out.squeeze(0),gold_lab)



    def predict(self, df):
        """ Method to evaluate a split.
        Inputs
        -----------------
        df - pd.DataFrame or dict. The data frame to be evaluated. This needs to 
                have a inp_tok key containing the toeknized input. It also 
                contains a label key containing the gold labels (integer-mapped).
                Also, a 'inp_ind' if words are split in characters/sub-words
        """
        inp = torch.Tensor(df['inp_tok']).to(torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.forward_pass(inp, df['inp_ind'])
            preds = torch.argmax(out,-1)
            
        return preds,out
                

    
        
