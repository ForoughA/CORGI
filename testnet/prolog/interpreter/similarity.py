from prolog.interpreter.term import Atom, NumberedVar, Number
from prolog.interpreter.memo import EnumerationMemo
from rpython.rlib.objectmodel import specialize
import numpy as np


def get_similarity_from_file(path, lambda_cut, entity_tnorm, predicate_tnorm):
    similarity = Similarity(lambda_cut, entity_tnorm, predicate_tnorm)
    with open(path) as f:
        text = f.read()
        for line in text.split('\n'):
            split = line.split('=')
            if len(split) < 2:
                continue
            split2 = split[0].split('~')
            
            word0 = split2[0].strip()
            word1 = split2[1].strip()
            # EDIT: add to simdict
            if word0 in similarity.simdict:
                similarity.simdict[word0].append(word1)
            else:
                similarity.simdict[word0] = [word1] 
            
            if word1 in similarity.simdict:
                similarity.simdict[word1].append(word0)
            else:
                similarity.simdict[word1] = [word0] 

            similarity.set_score(word0, word1, float(split[1].strip()))

    return similarity

def ruleterm_to_key(ruleterm):
    memo = EnumerationMemo()
    term = ruleterm.enumerate_vars(memo)
    memo.assign_numbers()
    return term_to_key(term)

def term_to_key(term):
    #print(term)
    if isinstance(term, Atom):
       return term.name()
    elif isinstance(term, NumberedVar):
         return ''
    #    return 'var' + str(term.num)
    else:
        result = term.signature().name
        for arg in term.arguments():
            if isinstance(arg, Number): #was failing on Numbers
                result += str(arg.num)
            else:
                result += term_to_key(arg)

        #print "result: ", result
        return result


class Similarity(object):
    def __init__(self, threshold, entity_tnorm='prod', predicate_tnorm='prod', var2index = None, index2var = None, embeddings = None, vectors = None):
        self._table = {}
        self._domain = {}
        self.lambda_cut = threshold
        self.threshold = threshold
        self.entity_tnorm_name = entity_tnorm
        self.predicate_tnorm_name = predicate_tnorm
        self.rule_scores = {}
        self.query_idx = 0

        # EDIT
        self.simdict = {}

        self.embeddings = embeddings
        self.var2index = var2index
        self.index2var = index2var
        self.vectors = vectors

    def _get_key(self, name1, name2):
        if name1 > name2:
            key = (name1, name2)
        else:
            key = (name2, name1)

        return key

    def parse_rulescores(self, text, engine):
        rules = []
        for line in text.split('\n'):
            # made a change here because this was throwing errors on >=
            split3 = line.split('. =') 
            split3[0] += '.'
            if len(split3) < 2:
                continue
            rule = split3[0].strip()
            rules.append(rule)
            scores = []
            for score in split3[1].split('|'):
                scores.append(float(score))
            parse = engine.parse(rule)[0][0]
            ruleterm = engine._term_expand(parse) 
            key = ruleterm_to_key(ruleterm)
            self.rule_scores[key] = scores

        engine.runstring("\n".join(rules), similarity=self)

    def get_score(self, name1, name2):
        if name1 == name2:
            return 1.0
        else:
            return self._table.get(self._get_key(name1, name2), 0)

    def get_score_signatures(self, signature1, signature2):
        if signature1.numargs != signature2.numargs:
            return 0
        else:
            return self.get_score(signature1.name, signature2.name)

    #TODO: change this to numpy
    def sim(self, embedding1, embedding2):
        def cosine_similarity(a, b):
            dot = np.dot(a, b)
            norma = np.linalg.norm(a)
            normb = np.linalg.norm(b)
            cos = dot / (norma * normb + 1e-8)
            return cos
        
        try:
            sims = (cosine_similarity(embedding1, embedding2) + 1.0)/2.0
        except:
            sims = 0
        return sims

    def compute_score(self, name1, name2):
        name1 = name1.strip()
        name2 = name2.strip()

        if name1 == name2:
            return 1.0
        else:
            
            if name1 in self.var2index:
                index1 = self.var2index[name1]
                embedding1 = self.embeddings[index1, :]
            else:
                return 0.0
            if name2 in self.var2index:
                index2 = self.var2index[name2]
                embedding2 = self.embeddings[index2, :] 
            else:
                return 0.0
            score = self.sim(embedding1, embedding2)
            
            if (score > 0.9):
                return 1.0 
            else:
                return 0.0
            #print('\n')"""

        return score


    def get_score_embeddings(self, signature1, signature2):
        if isinstance(signature2, str):
            if signature1.numargs != 0:
                return 0
            return self.compute_score(signature1.name, signature2)

        if signature1.numargs != signature2.numargs:
            return 0
        else: 
            return self.compute_score(signature1.name, signature2.name)

    def get_initial_rule_scores(self, ruleterm):
        key = ruleterm_to_key(ruleterm)
        if key not in self.rule_scores:
            print "Could not find ", key
        return self.rule_scores[key]

    def set_score(self, name1, name2, score):
        self._domain[name1] = None
        self._domain[name2] = None
        self._table[self._get_key(name1, name2)] = score

    def get_similar(self, name):
        similar = []
        for other in self._domain.keys():
            score = self.get_score(name, other)
            if score >= self.threshold:
                similar.append((other, score))
        # POSSIBLY SORT THE LIST BY SCORES
        return similar

    def reset_threshold(self):
        self.threshold = self.lambda_cut

    def function_tnorm(self, a, b):
        return self.tnorm(a,b, self.predicate_tnorm_name)

    def term_tnorm(self, a, b):
        return self.tnorm(a,b, self.entity_tnorm_name)

    def tnorm(self, a, b, name):
        if name == 'prod':
            return a*b
        elif name == 'luk':
            return max(0, a+b-1)
        elif name == 'min':
            return min(a, b)
        else:
            raise ValueError("Invalid t-norm " + name)

