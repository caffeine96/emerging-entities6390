import torch

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
        self.lstm = torch.nn.LSTM(emb_size, hidden_size,batch_first=True,\
                            dropout=dropout, bidirectional=True)
        # This is the classification head on the LSTM layer
        self.fc1 = torch.nn.Linear(hidden_size*2,int(hidden_size/2))
        self.relu = torch.nn.ReLU() 
        self.fc2 = torch.nn.Linear(int(hidden_size/2),len(data.labels))

        self.softmax = torch.nn.Softmax(dim=-1)

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
                    sent_idx.append(self.vocab.index(charac))
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


    def compute_loss(self, data):
        """ Compute Loss for the training sample
        Inputs
        ---------------
        data- dict. Dictionary containing the training sample information
        """
        gold_lab = torch.Tensor(data['label']).to(torch.long).to(self.device)
        inp = torch.Tensor(data['inp_tok']).to(torch.long).unsqueeze(0).to(self.device)
        emb_inp = self.embedding_layer(inp)
        lstm_out, _= self.lstm(emb_inp)
        lstm_out_comb = self._mean_pool(lstm_out,data['inp_ind'])
        out = self.fc2(self.relu(self.fc1(lstm_out_comb)))
        return self.loss_fn(out.squeeze(0),gold_lab)
        
        

