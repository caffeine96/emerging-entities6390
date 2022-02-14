from bilstmcrf import BiLSTMCRF

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
            self.model = BiLSTMCRF(data)
            self.model.prep_tokenizer(data)


    
    
    def train(self):
        tr_inp_tok, tr_inp_ind = self.model.tokenize(self.data.train_data)
        
