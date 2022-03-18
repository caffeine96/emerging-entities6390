import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(42)


class RoBERTa(torch.nn.Module):
    def __init__(self, data,  emb_size=768, crf_layer=False, device="cpu"):
        """ Constructor for the RoBERTa module
        Inputs
        ---------------
        data - EntData. Instance with processed inputs
        emb_size - int. Embedding Dimensions
        crf_layer - bool. Add the CRF layer
        device - str. Device on which computations must be made.
                    "cpu" or "cuda". Default-"cpu"
        """
        super(RoBERTa, self).__init__()
        self.emb_size = emb_size
        self.crf_layer = crf_layer
        self.device = device

        self.model = AutoModel.from_pretrained("roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

        self.fc1 = torch.nn.Linear(self.emb_size, int(emb_size/2))
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(int(emb_size/2),len(data.labels))

        self.softmax = torch.nn.Softmax(dim=-1)
        self.prep_loss(data)



    def prep_loss(self,data):
        """Prepares weighted loss
        """
        if not self.crf_layer:
            count_list = [0]*len(data.labels)
            total = 0
            for sent_lab in data.train_data['label']:
                for lab in sent_lab:
                    total+=1
                    count_list[lab] += 1

            class_weights = torch.Tensor(list(map(lambda x:1-(x/total),count_list)))
            print(f"\nClass Weights: {class_weights}")
            self.loss_fn= torch.nn.CrossEntropyLoss(weight=class_weights).to(self.device)



    def tokenize(self, inp_data):
        """ Tokenizes words down to sub-words
        Inputs
        ---------------
        inp_data - List. List of tokenized sentences

        Outputs
        --------------
        inp_tok - List . List of tokenized setences
        inp_ind - List . List of indices mapping subwords to their
                        respective words
        """
        inp_tok = []
        inp_ind = []

        for item in inp_data:
            tok = self.tokenizer(item, is_split_into_words=True,return_tensors='pt',add_special_tokens=False)
            inp_tok.append(tok['input_ids'][0])
            sent_pos = []
            prev = 0
            pos = []
            for ix, word_id in enumerate(tok.word_ids()):
                if word_id != prev:
                    prev = word_id
                    sent_pos.append(pos)
                    pos = []
                pos.append(ix)
            sent_pos.append(pos)
            
            inp_ind.append(sent_pos)
        
        return inp_tok, inp_ind



    def _mean_pool(self, inp, ixs):
        mean_pooled_emb = None
        for ix_set in ixs:
            avg_pooled = torch.mean(inp[0,ix_set,:], dim=0,keepdim=True)
            if mean_pooled_emb == None:
                mean_pooled_emb = avg_pooled
            else:
                mean_pooled_emb = torch.cat((mean_pooled_emb, avg_pooled) , dim=0)
        
        return mean_pooled_emb.unsqueeze(0)



    def forward_pass(self, inp, subword_map, print_log=False):
        """ Forward pass through the network
        Inputs
        ----------------
        inp - torch.Tensor. Input data
        subword_map - List. List mapping sub-words to words
        print_log- bool. Debugging flag
        """
        out = self.model(inp)
        out_hidd = out.last_hidden_state
        mean_pooled = self._mean_pool(out_hidd, subword_map)
        out_final = self.fc2(self.relu(self.fc1(mean_pooled)))

        return out_final



    def compute_loss(self, data, print_log=False):
        """ Compute Loss for the training sample
        Inputs
        ----------------
        data - dict. Dictionary containing the training sample information
        print_log - bool. Debgging flag
        """
        gold_lab = torch.Tensor(data['label']).to(torch.long).to(self.device)
        inp = data['inp_tok'].to(torch.long).unsqueeze(0).to(self.device)
        out_net = self.forward_pass(inp, data['inp_ind'], print_log)

        return self.loss_fn(out_net.squeeze(0),gold_lab)



    def predict(self, d_point):
        """ Method to evaluate a data point.
        Inputs
        -----------
        d_point - dict. Point to be evaluated.
        """
        inp = d_point['inp_tok'].to(torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.forward_pass(inp, d_point['inp_ind'])
            preds = torch.argmax(out,-1)

        return preds, out
