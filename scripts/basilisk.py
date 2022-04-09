import ast
import math
import pandas as pd
from data import EntData
from tqdm import tqdm
from reject_list import reject_list

DATA_DIR = "./../emerging_entities_17"


def extract_task_entities(data,category,rel_lab_ids, extract_contexts=False,c_wind=2):
    entities = []
    contexts = {}
    #print(rel_lab_ids)
    
    for ix in range(len(data['inp'])):
        sent = data['inp'][ix]
        lab = data['label'][ix]
        entity=[]
        context = []
        
        for ix_sent, word in enumerate(sent):
            if lab[ix_sent] == rel_lab_ids[0]:
                if len(entity) != 0:
                    phrase = " ".join(entity)
                    if phrase.lower() not in entities:
                        entities.append(phrase.lower())
                    
                    #Compute following context
                    #for previous entity
                    if extract_contexts:
                        if ix_sent+c_wind-1 < len(sent):
                            follow_ctxt = sent[ix_sent:ix_sent+c_wind]
                        else:
                            follow_ctxt = sent[ix_sent:]
                            follow_ctxt = follow_ctxt + [None]*(c_wind-len(follow_ctxt))
                        context.append(follow_ctxt)
                        
                        if context != [[None]*c_wind,[None]*c_wind]:
                            if str(context) not in contexts.keys():
                                contexts[str(context)] = []
                            if phrase not in contexts[str(context)]:
                                contexts[str(context)].append(phrase)
                # Compute preceding context for new entity
                if extract_contexts:         
                    if ix_sent-c_wind >= 0:
                        prv_ctxt = sent[ix_sent-c_wind:ix_sent]
                    else:
                        prv_ctxt = sent[:ix_sent]
                        prv_ctxt = [None]*(c_wind-len(prv_ctxt)) + prv_ctxt
                
                    context = [prv_ctxt] 

                entity = [word]
            
            elif lab[ix_sent] == rel_lab_ids[1]:
                entity.append(word)
            else:
                if len(entity) != 0:
                    phrase = " ".join(entity)
                    if phrase.lower() not in entities:
                        entities.append(phrase.lower())
                    #Compute following context
                    #for previous entity
                    if extract_contexts:
                        if ix_sent+c_wind-1 < len(sent):
                            follow_ctxt = sent[ix_sent:ix_sent+c_wind]
                        else:
                            follow_ctxt = sent[ix_sent:]
                            follow_ctxt = follow_ctxt + [None]*(c_wind-len(follow_ctxt))
                        context.append(follow_ctxt)
                        
                        if context != [[None]*c_wind,[None]*c_wind]:
                            if str(context) not in contexts.keys():
                                contexts[str(context)] = []
                            if phrase not in contexts[str(context)]:
                                contexts[str(context)].append(phrase)
                entity = []
 

            
        if len(entity) != 0:
            phrase = " ".join(entity)
            if phrase.lower() not in entities:
                entities.append(phrase.lower())
            if extract_contexts:
                context.append([None,None])
            
            if context != [[None]*c_wind,[None]*c_wind]:
                if str(context) not in contexts.keys():
                    contexts[str(context)] = []
                if phrase not in contexts[str(context)]:
                    contexts[str(context)].append(phrase)


    return entities, contexts
                    


def check_entity(ent, tweet):
    ent_list = ent.split()
    tweet_list = tweet.split()
    
    flag = False
    ids = []

    for ent_word in ent_list:
        if ent_word not in tweet_list:
            return False, ids

    for tw_ix,word in enumerate(tweet_list):
        if word == ent_list[0]:
            break_call = False
            if len(ent_list) > 1:
                try:
                    for fwd_ix, ent_word in enumerate(ent_list[1:]):
                        if ent_list[fwd_ix+1] != tweet_list[tw_ix+fwd_ix+1]:
                            break_call = True
                            break
                    if not break_call:
                        flag = True
                        ids.append(tw_ix)
                except IndexError:
                    pass
            else:
                flag = True
                ids.append(tw_ix)

    return flag, ids



def extract_context(tweet,idx,ent_len,c_wind):
    sent = tweet.lower().split()
    if idx-c_wind >= 0:
        prv_ctxt = sent[idx-c_wind:idx]
    else:
        prv_ctxt = sent[:idx]
        prv_ctxt = [None]*(c_wind-len(prv_ctxt)) + prv_ctxt
    
    back_id = idx+ ent_len

    if back_id + c_wind <= len(sent):
        fllw_ctxt = sent[back_id : back_id+c_wind]
    elif back_id == len(sent):
        fllw_ctxt = [None, None]
    else:
        fllw_ctxt = sent[back_id:]
        fllw_ctxt = fllw_ctxt + [None]*(c_wind-len(fllw_ctxt))

    return [prv_ctxt, fllw_ctxt]    





def extract_contexts_unlab(data_unlab, train_entities,ctxts,c_wind):
    for ix, tweet in tqdm(data_unlab.iteritems()):
        for ent in train_entities:
            flag,ids = check_entity(ent, tweet.lower())
            if flag:
                for idx in ids:
                    ctxt = extract_context(tweet,idx,len(ent.split()),c_wind)
                    
                    if ctxt != [[None]*c_wind,[None]*c_wind]:
                        if str(ctxt) not in ctxts.keys():
                            ctxts[str(ctxt)] = []
                        if ent not in ctxts[str(ctxt)]:
                            ctxts[str(ctxt)].append(ent)
                    

    return ctxts




def check_context(ctxt, tweet, thresh=5):
    sent = tweet.lower().split()
    prv_ctxt = ctxt[0]
    flw_ctxt = ctxt[1] 

    sent = [None]*len(prv_ctxt) + sent + [None]*len(flw_ctxt)
    entities = []
    
    for wrd in prv_ctxt:
        if wrd not in sent:
            return entities
    for wrd in flw_ctxt:
        if wrd not in sent:
            return entities

    for ix, word in enumerate(sent):
        if word == prv_ctxt[0]:
            match=True
            
            if len(prv_ctxt) > 1:
                if prv_ctxt != sent[ix: ix+len(prv_ctxt)]:
                    match = False

            if match:
                pot_ent = []
                new_ix = ix+len(prv_ctxt)
                thrsh_cnter = 0
                match = False
                for ix_rem, rem_word in enumerate(sent[new_ix:]):
                    if rem_word == flw_ctxt[0]:
                        if flw_ctxt == sent[new_ix+ix_rem:new_ix+ix_rem+len(flw_ctxt)]:
                                match = True
                                break
                        
                    if thrsh_cnter == thresh:
                        break
                    else:
                        thrsh_cnter += 1
                        pot_ent.append(rem_word)

            if match and len(pot_ent)>0:
                entities.append(" ".join(pot_ent))

    return entities


def extract_entities_unlab(data_unlab, seed_entities, ctxts, c_wind): 
    non_seed_ctxts = {ctxt_key:[] for ctxt_key in ctxts.keys()}

    for ix, tweet in tqdm(data_unlab.iteritems()):
        for ctxt in ctxts.keys():
            context = ast.literal_eval(ctxt)
            new_entities = check_context(context, tweet, thresh=3)
            for ent in new_entities:
                if (ent not in seed_entities) and \
                   (ent not in non_seed_ctxts[ctxt]):
                    non_seed_ctxts[ctxt].append(ent)
        
    return non_seed_ctxts


def calculate_rlogf(pattern_pool):
    for key in pattern_pool.keys():
        pos = pattern_pool[key]["pos_extr"]
        pattern_pool[key]["rlogf"] = (pos*math.log2(pos))/(pos+pattern_pool[key]["neg_extr"])

    return pattern_pool


def calculate_avglogscore(candidates, pattern_pool, new_extraction_ctxts):
    entity_scores = {}
    for ent in candidates:
        patterns_extracted = 0
        numerator = 0
        for key in pattern_pool.keys():
            if ent in new_extraction_ctxts[key]:
                patterns_extracted += 1
                numerator += math.log2(pattern_pool[key]["pos_extr"] + 1)
                #numerator = pattern_pool[key]["pos_extr"]
        entity_scores[ent] = numerator/patterns_extracted

    return entity_scores



def save_to_file(data_unlab, entities,category):
    for ent in entities:
        for ix, tweet in tqdm(data_unlab.iteritems()):
            flag,ids = check_entity(ent, tweet.lower())
            if flag:
                tweet_list = tweet.split()
                lab = ["O"]*len(tweet_list)
                for idx in ids:
                    lab[idx] = f"B-{category}"
                    for rem_ix in range(len(ent.split())-1):
                        lad[idx_rem_ix+1] = f"I-{category}"
                
                with open(f"{category}_aug.txt","a+") as f:
                    for tw_ix in range(len(tweet_list)):
                        f.write(f"{tweet_list[tw_ix]}\t{lab[tw_ix]}\n")
                    f.write("\n")
                    break






if __name__ == "__main__":
    train_file = f"{DATA_DIR}/wnut17train.conll"
    dev_file = f"{DATA_DIR}/emerging.dev.conll"
    test_file = f"{DATA_DIR}/emerging.test.annotated"
    data = EntData(train_file,dev_file,test_file)

    category = "creative-work"
    context_window = 2
    max_iter = 1
    init_pattern_pool = 3
    increment_pattern = 2
     
    rev_map = [data.labels.index(f"B-{category}"),data.labels.index(f"I-{category}")]

    og_train_entities, ctxts = extract_task_entities(data.train_data, category=category, rel_lab_ids=rev_map, extract_contexts=True,c_wind=context_window)
    dev_entities, _ = extract_task_entities(data.dev_data, category=category, rel_lab_ids=rev_map)
    test_entities, _ = extract_task_entities(data.test_data,category=category, rel_lab_ids=rev_map)

    data_unlab = pd.read_csv("./../../sentiment140.csv", engine ='python', error_bad_lines=False, header=0, names = ["0","1","2","3","4","tweet"])
    
    data_unlab = data_unlab['tweet']#.head(100000)

    seed_entities_pre = list(filter(lambda i: i not in reject_list[category], og_train_entities))
    
    discovered_entities = []

    for it in range(max_iter):
        pattern_pool = {}
        seed_entities = seed_entities_pre + discovered_entities 
        # Extract Seed entity contexts
        new_ctxts = extract_contexts_unlab(data_unlab,seed_entities,ctxts,c_wind=context_window)
      
        # Ensures singly triggered contexts are not considered
        top_ctxts = {key: new_ctxts[key] for key in new_ctxts.keys() if len(new_ctxts[key]) > 1}
        #top_ctxts = dict(sorted(new_ctxts.items(),key=lambda item: len(item),reverse=True)[:top_cont])

        for key in top_ctxts.keys():
            pattern_pool[key] = {"pos_extr": len(top_ctxts[key])}

        # Extarct Non-seed entity contexts
        ctxts_neg = extract_entities_unlab(data_unlab, seed_entities, top_ctxts, c_wind=context_window)
        for key in top_ctxts.keys():
            pattern_pool[key]["neg_extr"] = len(ctxts_neg[key])
        
        # This contains all the patterns with at least 2 extarction
        pattern_pool = calculate_rlogf(pattern_pool)

        pattern_pool_size = init_pattern_pool + increment_pattern*it

        # Note that here we have used a different notation.
        # This technically is the pattern_pool from which candidates will be extracted
        rel_pattern_pool = dict(sorted(pattern_pool.items(), key=lambda item: item[1]["rlogf"], reverse=True)[:pattern_pool_size])
        
        candidate_entities = []
        for key in rel_pattern_pool:
            for en in ctxts_neg[key]:
                if (en not in candidate_entities) or (en not in dev_entities) or (en not in test_entities):
                    candidate_entities.append(en)

        entity_scores = calculate_avglogscore(candidate_entities,pattern_pool,ctxts_neg)
        entities_to_add = dict(sorted(entity_scores.items(), key=lambda item: item[1], reverse=True)[:10])
        print(f"Entities to add: {entities_to_add.keys()}")
        for key in entities_to_add.keys():
            discovered_entities.append(key)
    
    save_to_file(data_unlab, discovered_entities, category)
