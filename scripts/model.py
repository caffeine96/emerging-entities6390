from datetime import datetime
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


    
    
    def train(self, max_ep=100, lr=0.001, save_model=False):
        # Tokenize and convert words into sub-words/characters 
        # and sub-words/characters to indices
        ctime = datetime.now()
        folder_name = f"{ctime.month}{ctime.day}_{ctime.hour}{ctime.minute}/"
        if save_model:
            Path.mkdir(Path(MODEL_DIR+folder_name), exist_ok=True, parents=True)
            with open("log.txt","a+") as f:
                f.write(f"Experiment Name- {self.exp_name}")
                f.write(f"Learning Rate- {lr}")
                
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

            if dev_metrics['total_ent']['f1'] > best_metric and save_model:
                best_metric = dev_metrics['total']['f1']
            
                torch.save({
                    'epoch' : ep,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'metrics': dev_metrics
                    }, f"{MODEL_DIR}{folder_name}{ep}")


    def evaluate(self, df, save=True, best_metric=0, course_correction=True):
        all_pred = []
        all_gold = []
        all_word = []
        all_gold_sent =[]
        all_pred_sent =[]
        all_word_sent =[]
        
        violations = 0
        for ix, row in tqdm(df.iterrows()):
            preds ,out = self.model.predict(row)
            if course_correction:
                preds, violations_new = self.course_correction(preds,out)
                violations += violations_new
            all_pred.extend(preds[0].tolist())
            all_gold.extend(row['label'])
            all_word.extend(row['inp'])
            all_pred_sent.append(preds[0].tolist())
            all_gold_sent.append(row['label'])
            all_word_sent.append(row['inp'])
        print(f"Illegal Predicitons :{violations}")
    
        eval_metrics = self.calc_f1(all_gold,all_pred)
        eval_metrics_entity = self.calc_f1_ent(all_gold_sent, all_pred_sent, all_word_sent)
        # rename dictionary
        for key in eval_metrics_entity.keys():
            eval_metrics[key+"_ent"] = eval_metrics_entity[key]

        print(f"""Dev Metric: {eval_metrics['total_ent']['f1']}\n""")
        

        if save and (eval_metrics['total_ent']['f1']>best_metric):
            with open("dev_preds.txt","w+") as f:
                for ix in range(len(all_gold)):
                    g_lab = self.data.labels[all_gold[ix]]
                    p_lab = self.data.labels[all_pred[ix]]
                    line = f"{all_word[ix]}\t{g_lab}\t{p_lab}\n"
                    f.write(line)
        return eval_metrics




    def course_correction(self,pred, scores):
        corrected = []
        violations = 0
        for ix in range(pred.shape[0]):
            pred_sent = pred[ix]
            scores_sent = scores[ix]
            corrected_sent = []
            for ix_sent in range(pred_sent.shape[0]):
                curr = int(pred_sent[ix_sent])
                # For the first word
                try:
                    prev = int(corrected_sent[ix_sent-1])
                except IndexError:
                    prev = 0

                if self.data.valid_set[curr][prev] == 1:
                    # No violation
                    corrected_sent.append(int(pred_sent[ix_sent].item()))
                else:
                    # Violation. Iterate over other labels in descending
                    # order of score 
                    score_idx = scores_sent[ix_sent]        
                    next_best = torch.argsort(score_idx, descending=True)[1:]
                    for alt_ix in range(next_best.shape[0]):
                        if self.data.valid_set[next_best[alt_ix]][prev] == 1:
                            corrected_sent.append(int(next_best[alt_ix].item()))
                            break
                    violations += 1
            
            corrected.append(corrected_sent)

        return torch.Tensor(corrected).to(torch.int), violations





    def load_model(self,model_path):
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
                





    def calc_f1(self, gold, preds):
        conv_gold = [self.data.labels[el] for el in gold]
        conv_pred = [self.data.labels[el] for el in preds]
    
        cf = {
                "total": np.zeros((2,2)),
                "corporation": np.zeros((2,2)),
                "person"  : np.zeros((2,2)),
                "creative-work"  : np.zeros((2,2)),
                "group": np.zeros((2,2)),
                "location": np.zeros((2,2)),
                "product": np.zeros((2,2)),
                "B": np.zeros((2,2)),
                "I": np.zeros((2,2)),
                "O": np.zeros((2,2))
        }
        
        for ix in range(len(conv_gold)):
            g_lab = conv_gold[ix]
            p_lab = conv_pred[ix]
            # Increases confusion matrix count by category
            # and total (Micro)
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


            # Compute Unatatched BIO conf matrix
            # for unattached F1 score
            if g_lab!='O':
                g_bio = g_lab.split('-')[0]
            else:
                g_bio = g_lab

            if p_lab!='O':
                p_bio = p_lab.split('-')[0]
            else:
                p_bio = p_lab
        
            if g_bio == p_bio:
                cf[g_bio][0][0] += 1
            else:
                cf[g_bio][0][1] += 1
                cf[p_bio][1][0] += 1
        
        metrics = self.calc_f1_from_cf(cf)
        
        return metrics



    def calc_f1_from_cf(self,cf):
        # Calculate F1
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




    def calc_f1_ent(self, gold, pred, sents): 
        cf_ent = {
                "total": np.zeros((2,2)),
                "corporation": np.zeros((2,2)),
                "person"  : np.zeros((2,2)),
                "creative-work"  : np.zeros((2,2)),
                "group": np.zeros((2,2)),
                "location": np.zeros((2,2)),
                "product": np.zeros((2,2)),
                "extr": np.zeros((2,2))
        }

        def extract_ent(words, lab):
            entities = []
            for ix in range(len(words)):
                if lab[ix].split("-")[0] == "B":
                    entity = {'word':words[ix],'lab':"-".join(lab[ix].split("-")[1:])}
                    if ix == len(words) -1:
                        entities.append(entity)
                    for cont_ix in range(ix+1,len(words)):
                        if lab[cont_ix].split("-")[0] in ["B","O"]:
                            entities.append(entity)
                            break
                        else:
                            entity['word'] += f" {words[cont_ix]}"

                        if cont_ix == len(words)-1:
                            entities.append(entity)
            return entities


        for sent_id in range(len(sents)):
            sent = sents[sent_id]
            conv_gold = [self.data.labels[el] for el in gold[sent_id]]
            conv_pred = [self.data.labels[el] for el in pred[sent_id]]
            
            extr_gold = extract_ent(sent,conv_gold)
            extr_pred = extract_ent(sent,conv_pred)
            
            for g_ent in extr_gold:
                flag=False
                for p_ent in extr_pred:
                    # If entity is correctly extarcted
                    if g_ent["word"] == p_ent["word"]:
                        cf_ent["extr"][0][0] += 1   # Correct extarction TP
                        flag = True # We will have processed this entity
                        if g_ent["lab"] == p_ent["lab"]:
                            # Correct extarction and label TP
                            cf_ent[g_ent["lab"]][0][0] += 1
                            cf_ent["total"][0][0] += 1
                        else:
                            # Correct extraction but incorr. label FN
                            cf_ent[g_ent["lab"]][0][1] += 1
                            cf_ent["total"][0][1] += 1
                            # False positive for the predicted category
                            cf_ent[p_ent["lab"]][1][0] += 1
                            cf_ent["total"][1][0] += 1
                    # If processed, we are done
                    if flag:
                        break
                # If not processed then FN
                if not flag:
                    cf_ent[g_ent["lab"]][0][1] += 1
                    cf_ent["total"][0][1] += 1
                    cf_ent["extr"][0][1] += 1


            # Adding remaining FPs
            for p_ent in extr_pred:
                flag=False
                for g_ent in extr_gold:
                    if g_ent["word"] == p_ent["word"]:
                        flag = True
                        break
                if not flag:
                    cf_ent[p_ent["lab"]][1][0] += 1
                    cf_ent["total"][1][0] += 1
                    cf_ent["extr"][1][0] += 1
        
        metrics = self.calc_f1_from_cf(cf_ent)
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
