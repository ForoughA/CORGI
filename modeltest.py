import torch
from net import Net
import sys
import pickle
import numpy as np
import ast

def handleNone(string):
    if (string.strip() == 'None'):
        return None
    return string
def main():
    query = sys.argv[1]
    parent_rule = handleNone(sys.argv[2])
    sister_rule = handleNone(sys.argv[3])
    var = sys.argv[3:]
    

    # load hidden state of charRNN if necessary

    state = torch.load('model_testing-reasoning-17.tch')
    state_dict = state['state_dict']
    config = state['config']
    

    model = Net(config, load_embeddings = False)
    model.load_state_dict(state_dict)
    model.eval() 

    query = model.query_to_tensor(query)

    if (parent_rule is not None):
        parent_rule = model.rule_to_tensor(parent_rule)
    if (sister_rule is not None):
        sister_rule = model.rule_to_tensor(sister_rule)

    rule = (parent_rule, sister_rule)
    #if (var is not None):
    var = model.vars_to_tensor(None)
        

    rule_dist, term_dist, var_dist = model(query, rule, var)

    rule_dist = rule_dist.detach().numpy()[0][0]
    term_dist = term_dist.detach().numpy()[0][0]
    #var_dist = [v.detach().numpy()[0][0] for v in var_dist]

    results = (rule_dist, term_dist)

    """print('RULEDIST')
    print(np.exp(rule_dist))
    print('VARDIST')
    print(var_dist)
    print('TERMDIST')
    print(np.exp(term_dist))"""
    with open('model_output.pkl', 'wb+') as f:
        pickle.dump(results, f, protocol = 2)

if __name__ == "__main__":
    main()

