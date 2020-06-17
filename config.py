
class Config(object):
    def __init__(self):
                
        # for rulehead character RNN
        self.char2index = {}
        self.index2char = {}
        self.n_chars = 0

        # for rule embeddings 
        self.rule2index = {}
        self.index2rule = {}
        self.n_rules = 0

        #for variable embeddings 
        self.var2index = {'None' : 0}
        self.index2var = {0 : 'None'}
        self.n_vars = 1 


        # configuration info for NN
        self.allowed_keys = {'CHAR_DIM', 'RULE_DIM', 'VAR_DIM', 
                        'HIDDEN_DIM', 'HIDDEN_CHAR_DIM', 
                        'RULEKEY_DIM', 'STATE_DIM', 
                        'DROPOUT_P', 'CHARSET_SIZE',
                        'VARSET_SIZE', 'RULESET_SIZE'}
        self.info = {'CHAR_DIM' : 256, 'RULE_DIM' : 256, 'VAR_DIM' : 300,
                     'HIDDEN_DIM' : 256, 'HIDDEN_CHAR_DIM' : 256, 'RULEKEY_DIM' : 128,
                     'STATE_DIM' : 256, 'DROPOUT_P' : 0.1, 'RULESET_SIZE' : 0, 'CHARSET_SIZE' : 0, 'VARSET_SIZE' : 0, 'MAX_VARS' : 4}

    # add rules and characters

    def isFact(self, rule):
        """ rule : a prolog fact/rule as a string
            returns a boolean value indicating whether input is a fact"""

        if (len(rule.split(':-')) < 2):
            return True
        else:
            return False

    def getFactSurface(self, fact):
        """ fact : prolog fact as a string
            returns string in the form <predicate>/<arity>"""

        args = self.getVars(fact)
        endPred = fact.find('(')
        predicate = fact[:endPred]
        fact = predicate + '/' + str(len(args))
        return fact


    def addRule(self, rule):
        #strip whitespace so rules with diff spacing are regarded
        #as the same rule 
        #rule = rule.replace(' ', '').lower()
        
        if not rule.strip() or '/*' in rule:
            return

        if self.isFact(rule):
            fact = self.getFactSurface(rule)
            if fact not in self.rule2index:
                self.rule2index[fact] = self.n_rules
                self.index2rule[self.n_rules] = fact
                self.n_rules += 1
                self.addChars(rule)

        else:
            if rule not in self.rule2index:
                self.rule2index[rule] = self.n_rules
                self.index2rule[self.n_rules] = rule
                self.n_rules += 1
                self.addChars(rule)
        
         
    def addChars(self, rule):
        chars = set(rule)
        for c in chars:
            if c not in self.char2index:
                self.char2index[c] = self.n_chars
                self.index2char[self.n_chars] = c
                self.n_chars += 1

    
    def addNums(self):
        for num in range(0, 24):
            if str(num) not in self.var2index:
                self.var2index[str(num)] = self.n_vars
                self.index2var[self.n_vars] = str(num)
                self.n_vars += 1

    def getVars(self, rule):
        endRuleHead = rule.find(':-')
        
        if (endRuleHead == -1):
            endRuleHead = rule.find('.')

        ruleHead = rule[:endRuleHead]

        startArgs = ruleHead.find('(') 

        if (startArgs == -1):
            return []

        startArgs = startArgs + 1
        endArgs = ruleHead.find(')')

        args = ruleHead[startArgs:endArgs]
        args = args.split(',')
        
        args = list(map(str.strip, args))

        return args

    
    def addVars(self, rule):
        args = self.getVars(rule)
        for arg in args:
            if arg not in self.var2index:
                self.var2index[arg] = self.n_vars
                self.index2var[self.n_vars] = arg
                self.n_vars += 1

    
    def addInfo(self, key, value):
        if key in self.allowed_keys: 
            self.info[key] = value


