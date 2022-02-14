import torch

class BiLSTMCRF(torch.nn.Module):
    def __init__(self, data, emb_size= 1024, hidden_size = 768, dropout=0.01):
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
        # Intializing the LSTM layer. Inputs are embedded character
        # vectors which will be initailized once we know the vocabulary of 
        # characters.
        self.lstm = torch.nn.LSTM(emb_size, hidden_size,batch_first=True,\
                            dropout=dropout, bidirectional=True)
        # This is the classification head on the LSTM layer
        self.fc1 = torch.nn.Linear(hidden_size*2,int(hidden_size/2))
        self.relu = torch.nn.ReLU() 
        self.fc2 = torch.nn.Linear(int(hidden_size/2),len(data.labels))



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
        self.embedding_layer = torch.nn.Embedding(len(vocab),self.emb_size)
        
    
    
    def tokenize(self, inp_data):
        inp_tok = []
        inp_ind = []

        for item in inp_data['inp']:
            sent_idx = []
            sent_pos = []
            for ix, word in enumerate(item):
                pos = []
                for charac in word:
                    sent_idx.append(charac)
                    pos.append(ix)
                sent_pos.append(pos)
            inp_tok.append(sent_idx)
            inp_ind.append(sent_pos)
            
            print(inp_tok)
            print(inp_ind)
            exit()


