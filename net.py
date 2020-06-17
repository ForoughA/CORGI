import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from char_rnn import CharRNN
from config import Config
from attention import Attention 
import numpy as np
import pickle


class Net(nn.Module):

    def __init__(self, config, load_embeddings = True):
        """
        config: object containing configuration info
        """
        super(Net, self).__init__()
    
        self.config = config
        
            
        # preprocessed sizes for embedding layers
        self.charset_size = config.info["CHARSET_SIZE"]
        self.ruleset_size = config.info["RULESET_SIZE"]
        self.varset_size = config.info["VARSET_SIZE"]
        
        # dimensions for embedding layers
        self.rule_embedding_dim = config.info["RULE_DIM"]
        self.var_embedding_dim = config.info["VAR_DIM"]
        self.char_embedding_dim = config.info["HIDDEN_CHAR_DIM"]
        self.state_embedding_dim = config.info["STATE_DIM"]
        self.key_embedding_dim = config.info["RULEKEY_DIM"]        
        
        # dimension of LSTM hidden vector
        self.hidden_dim = config.info["HIDDEN_DIM"]

        # dimension of character RNN hidden vector
        self.hidden_char_dim = config.info["HIDDEN_CHAR_DIM"]

        # dropout parameter
        self.dropout_p = config.info["DROPOUT_P"]

        # number of vars 
        self.maxvars = config.info["MAX_VARS"]
       
        # character RNN for query rulehead
        self.hidden_char = self.init_hidden(self.hidden_char_dim)
        self.char_embedding = nn.Embedding(self.charset_size, self.char_embedding_dim)
        self.char_lstm = nn.LSTM(self.char_embedding_dim, self.hidden_char_dim)

        # attention for variables 
        self.var_embedding = nn.Embedding(self.varset_size, self.var_embedding_dim) 
        self.var_attention = Attention(self.varset_size, self.var_embedding_dim)
        if (load_embeddings):
            self.glove = pickle.load(open('glove.pkl', 'rb'))
            self.init_variable_embeddings() #initializes using GloVe        

        # ENCODING layer (f_enc(q_t, v_t) = s_t)
        #self.enc_input_dim = self.hidden_char_dim + self.maxvars * self.var_embedding_dim
        #self.var_encoder = nn.Linear(self.maxvars * self.var_embedding_dim, 50)
        self.enc_input_dim = self.hidden_char_dim 
        self.enc_fc1 = nn.Linear(self.enc_input_dim, self.state_embedding_dim)
        self.enc_fc2 = nn.Linear(self.state_embedding_dim, self.state_embedding_dim)

               
        # LSTM layer (f_lstm)
        self.hidden = self.init_hidden(self.hidden_dim, nlayers = 2)
        self.lstm_input_dim = self.state_embedding_dim + self.rule_embedding_dim * 2
        self.lstm = nn.LSTM(self.lstm_input_dim, self.hidden_dim, 2)

        # RULE layers (f_rule)
        #self.rule_keys = nn.Embedding(self.ruleset_size, self.key_embedding_dim)
        
        # Rule Embedding Layer (M_rule)
        self.rule_embedding = nn.Embedding(self.ruleset_size, self.rule_embedding_dim)

        self.rule_fc1 = nn.Linear(self.hidden_dim, self.rule_embedding_dim)
        self.rule_fc2 = nn.Linear(self.rule_embedding_dim, self.rule_embedding_dim)  

        # Termination layers (f_end)
        
        self.term_fc = nn.Linear(self.hidden_dim, 2)

        # Variable layers (f_var)

        self.vars_fc1 = nn.Linear(self.var_embedding_dim, self.var_embedding_dim)
        self.vars_fc2 = nn.Linear(self.var_embedding_dim, self.varset_size)  


    def forward(self, query, rule, var):
        """ query: indexes corresponding to characters in query head in a LongTensor 
            
            rule: tuple with 2 LongTensors, the first contains the index of the parent rule and the 
            second contains the index of the sister rule
            
            var: LongTensor containing var_ids of variables of the rule

            returns a distrbution over the rules, a distribution over 
            termination statuses and a distribution over variables
        """
        q_t = self.get_query_embedding(query)
        parent_rule, sister_rule = rule
        if (parent_rule is not None):
            parent_t = self.rule_embedding(parent_rule)
        else:  
            parent_t = torch.zeros_like(self.rule_embedding(torch.LongTensor([0])))
        if (sister_rule is not None):
            sister_t = self.rule_embedding(sister_rule)
        else:  
            sister_t = torch.zeros_like(self.rule_embedding(torch.LongTensor([0])))
        r_t = torch.cat((parent_t.view(1, 1, -1), sister_t.view(1, 1, -1)), dim = 2)

        h = self.get_hidden(q_t, r_t) # [1, 1, 256]
        
        # get distribution over rulekeys
        key = self.rule_fc2(F.elu(self.rule_fc1(h))) # [1, 1, 256]
        rulekeys = self.rule_embedding.weight.t() # [256, num_rules]
        rule_distribution = F.log_softmax(torch.matmul(key, rulekeys), dim = 2) # [1, 1, num_rules]
        
        # get termination probability
        term = F.log_softmax(self.term_fc(h), dim = 2) # [1, 1, 2]

        # get variable distributions
        v_t = var 
        var_output = []

        for i in range(self.maxvars):
            v = v_t[i].view(1, 1, -1)
            var = F.log_softmax(self.vars_fc2(F.elu(self.vars_fc1(v))), dim = 2) # [1, 1, num_vars]
            var_output.append(var)

        return rule_distribution, term, var_output


    def query_to_tensor(self, query):
        """ query : query head as a string
            returns indexes corresponding to characters in a LongTensor """
        
        query = query.split('(')[0]
        chartensor = torch.zeros(len(query)).long()
        
        for i in range(len(query)):
            c = query[i]
            if c in self.config.char2index:
                chartensor[i] = self.config.char2index[c]
                
        return Variable(chartensor)


    def get_query_embedding(self, query):
        """ query : indices corresponding to characters in a LongTensor 
            returns character level embedding of query"""

        char_embeds = self.char_embedding(query)
        char_lvl, self.hidden_char = self.char_lstm(char_embeds.view(len(query),1,-1), self.hidden_char) 

        return char_lvl[-1]


    def rule_to_tensor(self, rule):
        """ rule : rule as a string
            returns index corresponding to rule in a LongTensor """
        
        try: 
            rule = rule.strip()
        
            ruletensor = torch.zeros(1).long()

            if (self.config.isFact(rule)):
                rule = self.config.getFactSurface(rule)

            ruletensor[0] = self.config.rule2index[rule]

            return Variable(ruletensor) 
        except:
            return None


    def vars_to_tensor(self, var):
        """ var : list of vars 
            returns indices corresponding to vars in a LongTensor"""
        if var is None:
            return torch.zeros((self.maxvars, 1, self.var_embedding_dim))
 
            
        vartensor = torch.zeros((len(var), 1, 300))
        
        for i in range(len(var)):
            v = str(var[i])

            if v in self.config.var2index:
                x = torch.zeros(1).long()
                x[0] = self.config.var2index[v]
                vartensor[i] = self.var_embedding(x)
            else:
                vartensor[i] = torch.zeros(1, 1, 300)

        return Variable(vartensor)


    """def get_vars_embedding(self, var):
        var : indices corresponding to variables in a LongTensor
            returns embedding of variables with attention
        if (var is not None):
            vartensor = self.var_embedding(var).view(len(var), 1, -1)
        else:
            vartensor = torch.zeros((self.maxvars, 1, self.var_embedding_dim))
        return vartensor"""


    def get_hidden(self, query, rule):
        """ query: character level embedding of query
            rule: embedding of rule at current step
        """
        """if (var is not None):
            var = F.elu(self.var_encoder(var.view(1, 1, -1)))
        merged = torch.cat([query.view(1, 1, -1), var.view(1, 1, -1)], dim = 2)"""

        merged = query.view(1, 1, -1)
        
        # pass concatenated q_t and v_t through f_enc to get s_t
        state = F.elu(self.enc_fc1(merged))
        state = self.enc_fc2(state)
        
        # concatenate state with rule embedding and pass through LSTM 
        merged = torch.cat([state.view(1, 1, -1), rule.view(1, 1, -1)], dim=2)
        
        out, self.hidden = self.lstm(merged, self.hidden)
        h_t = out[-1].view(1, 1, -1)
        
        # layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size)
        #h_t = h_t.view(2, -1, 1, self.hidden_dim)[-1]
        #print(h_t)
        return h_t


    def init_hidden(self, size, nlayers = 1, batch_size = 1):
        """ size : dimension of hidden vector 
            returns (h_0, c_0) for LSTM"""
        hidden = (Variable(torch.zeros(nlayers, batch_size, size)), 
                  Variable(torch.zeros(nlayers, batch_size, size)))
        return hidden   
    

    def init_variable_embeddings(self):
        emb_mean,emb_std = -0.005838499,0.48782197
        nb_words = self.config.n_vars
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, 300))

        for (i, var) in self.config.index2var.items():
            if var in self.glove:
                print(var)
                embedding_vector = self.glove[var]
                embedding_matrix[i] = embedding_vector

        self.var_embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

    
    def init_variable_embeddings2(self, file = "glove.840B.300d.pkl"):
        """ initializes variable embedding layer using GloVe 300d embeddings"""
        vardict = self.config.var2index 
        with open('glove.840B.300d.pkl') as glove:
            embeddings_index = pickle.load(glove)   
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file))
    
        #all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = -0.005838499,0.48782197
        #embed_size = all_embs.shape[1]

        nb_words = len(vardict)
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, 300))
            
        def vec(w):
            return embedding_matrix.loc[w].as_matrix()

        for word, i in vardict.items():
            print(word)
            embedding_vector = embeddings_index.get(word)

            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector

            else:
                embedding_vector = embeddings_index.get(word.capitalize())
                if embedding_vector is not None: 
                    embedding_matrix[i] = embedding_vector

        self.var_embedding.weight.data.copy_(torch.fromnumpy(embedding_matrix))

    
    def write_similarities(self, file = "./sim2.txt"):
        """calculates similarities from embeddings and write to file""" 
            
        def cosine_similarity(x1, x2=None, eps=1e-8):
            x2 = x1 if x2 is None else x2
            w1 = x1.norm(p=2, dim=1, keepdim=True)
            w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
            return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


        def similarity(x, y):
            sims = (cosine_similarity(x, y) + 1.0)/2.0
            return sims
        
        # write atom similarities
        vardict = self.config.index2var

        f = open(file, "w")

        for i in range(self.config.info["VARSET_SIZE"] - 1):
            var1 = vardict[i]
            #print(var1)
            if var1[0].isupper(): # pass if not atom
                continue

            emb1 = self.var_embedding(torch.LongTensor([i]))
            
            for j in range(i+1, self.config.info["VARSET_SIZE"]):
                
                var2 = vardict[j]
                #print(var2)
                if var2[0].isupper():
                    continue
                
                emb2 = self.var_embedding(torch.LongTensor([j]))
                sim = similarity(emb1, emb2)
                #sim = (0.5 * (1 + np.dot(emb1, emb2)
                #            / (np.linalg.norm(emb1) * np.linalg.norm(emb2).t()))[0]
                
                f.write("%s ~ %s = %f\n" % (var1, var2, sim))
                

        # write rule predicate similarities 
        # TODO: ???? How to deal with predicates that have multiple associated rules ????
        ruledict = self.config.index2rule

        def getPredicate(rule):
            endPred = rule.find("(")
            
            if (endPred == -1):
                endPred = rule.find(":")

            if (endPred == -1):
                endPred = rule.find(".")

            return rule[:endPred]

        for i in range(self.config.info["RULESET_SIZE"] - 1):

            rule1 = ruledict[i]
            pred1 = getPredicate(rule1)
            emb1 = self.rule_embedding(torch.LongTensor([i]))
 
            for j in range(i+1, self.config.info["RULESET_SIZE"]):
                rule2 = ruledict[j]
                pred2 = getPredicate(rule2)
                emb2 = self.rule_embedding(torch.LongTensor([j]))

                #sim = (0.5 * (1 + np.dot(emb1, emb2)
                #            / (np.linalg.norm(emb1) * np.linalg.norm(emb2).t()))[0]
                sim = similarity(emb1, emb2)
                f.write("%s ~ %s = %f\n" % (pred1, pred2, sim))

        f.close()


        


