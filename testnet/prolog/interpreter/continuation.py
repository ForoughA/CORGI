import py
import time
import pickle
import numpy as np
from subprocess import call
from rpython.rlib import jit
from rpython.rlib.objectmodel import we_are_translated, specialize
from prolog.interpreter import error
from prolog.interpreter import helper
from prolog.interpreter.term import Term, Atom, BindingVar, Callable, Var, Number
from prolog.interpreter.function import Function, Rule
from prolog.interpreter.heap import Heap
from prolog.interpreter.signature import Signature
from prolog.interpreter.module import Module, ModuleWrapper
from prolog.interpreter.helper import unwrap_predicate_indicator
from prolog.interpreter.stream import StreamWrapper
from prolog.interpreter.small_list import inline_small_list
from prolog.interpreter.similarity import Similarity

Signature.register_extr_attr("function", engine=True)

# ___________________________________________________________________
# model stuff

def run_model(query_string, rule_string, var_list):
    parent_rule, sister_rule = rule_string
    query_string = "'" + query_string + "'" 
    if (parent_rule is not None):
        parent_rule = "'" + parent_rule.source + "'"
    if (sister_rule is not None):
        sister_rule = "'" + sister_rule.source + "'"
    print 'query', query_string
    print 'parent', parent_rule
    print 'sister', sister_rule
    print '\n'
    v = var_list
    command = 'python3 modeltest.py %s %s %s %s %s %s %s' % (query_string, parent_rule, sister_rule, v[0], v[1], v[2], v[3])
    call(command, shell = True)

    with open('model_output.pkl', 'rb') as f:
        (rule_dist, term_dist) = pickle.load(f)

    # apply exp to squeeze logsoftmax from [-inf, 0] to [0, 1] 
    return np.exp(rule_dist), np.exp(term_dist)

def isFact(rule):
    return isinstance(rule, str)
        

# ___________________________________________________________________
# JIT stuff

def get_printable_location(rule, sconttype):
    if rule:
        s = rule.signature.string()
    else:
        s = "No rule"
    return "%s %s" % (s, sconttype)

def get_jitcell_at(where, rule):
    # XXX can be vastly simplified
    return rule.jit_cells.get(where, None)

def set_jitcell_at(newcell, where, rule):
    # XXX can be vastly simplified
    rule.jit_cells[where] = newcell

predsig = Signature.getsignature(":-", 2)
callsig = Signature.getsignature(":-", 1)

jitdriver = jit.JitDriver(
        greens=["rule", "sconttype"],
        reds=["scont", "fcont", "heap"],
        get_printable_location=get_printable_location,
        #get_jitcell_at=get_jitcell_at,
        #set_jitcell_at=set_jitcell_at,
        )

# ___________________________________________________________________
# end JIT stuff


def driver(scont, fcont, heap, engine):
    rule = None
    oldscont = None
    while not scont.is_done():
        sconttype = scont.cont_type_name
        #print "sconttype", sconttype
        if isinstance(scont, ContinuationWithRule):
            rule = scont.rule
        if isinstance(scont, RuleContinuation) and rule.body is not None:
            jitdriver.can_enter_jit(rule=rule, sconttype=sconttype, scont=scont, fcont=fcont,
                                    heap=heap)
        try:
            jitdriver.jit_merge_point(rule=rule, sconttype=sconttype, scont=scont, fcont=fcont,
                                      heap=heap)
            oldscont = scont
            scont, fcont, heap  = scont.activate(fcont, heap)
            
            if (scont is None):
                return

            assert heap is not None
        except error.UnificationFailed:
            if not we_are_translated():
                if fcont.is_done():
                    raise
            scont, fcont, heap = fcont.fail(heap)
        except error.CatchableError, exc:
            pass
            scont, fcont, heap = scont.engine.throw(exc, scont, fcont, heap, rule)
        # if not a RuleContinuation????
        else:
            scont, fcont, heap = _process_hooks(scont, fcont, heap, engine)
    assert isinstance(scont, DoneSuccessContinuation)

    if scont.failed:
        raise error.UnificationFailed

@jit.unroll_safe
def _process_hooks(scont, fcont, heap, engine):
    if heap.hook:
        e = engine
        #e = scont.engine
        hookcell = heap.hook
        heap.hook = None
        while hookcell:
            attvar = hookcell.attvar
            att = attvar.getbinding()
            attmap = jit.hint(attvar.attmap, promote=True)
            for i in range(len(attvar.value_list)):
                val = attvar.value_list[i]
                if val is None:
                    continue
                module = attmap.get_attname_at_index(i)
                query = Callable.build("attr_unify_hook", [val, att])
                try:
                    mod = e.modulewrapper.get_module(module, query)
                except error.CatchableError, err:
                    scont, fcont, heap = scont.engine.throw(err, scont, fcont, heap)
                    break
                scont, fcont, heap = e.call_in_module(query, mod, scont, fcont, heap)
                heap.add_trail_atts(attvar, module)
            hookcell = hookcell.next
            attvar.value_list = None # XXX?
    return scont, fcont, heap


class Engine(object):
    def __init__(self, load_system=False, similarity=None, max_depth=20, lambda_cut=0.1, embeddings = None, rule2index = None, maxk = 5):
        self.similarity = similarity or Similarity(lambda_cut)
        self.operations = None
        self.modulewrapper = ModuleWrapper(self)
        self.max_depth = max_depth
        if load_system:
            self.modulewrapper.init_system_module()
        
        from prolog.builtin.statistics import Clocks
        self.clocks = Clocks()
        self.clocks.startup()
        self.streamwrapper = StreamWrapper()
        self.embeddings = embeddings
        self.rule2index = rule2index
        self.maxk = maxk
        self.index2rule = {}


    def _freeze_(self):
        return True

    # _____________________________________________________
    # database functionality

    def add_rule(self, ruleterm, end=True):
        module = self.modulewrapper.current_module
        rule = self.make_rule(ruleterm, module)
        signature = rule.signature
        
        if self.get_builtin(signature):
            error.throw_permission_error(
                "modify", "static_procedure", rule.head.get_prolog_signature())

        function = module.lookup(signature) #adding to module.functions
        function.add_rule(rule, end) #adding to rulechain
        return rule

    def make_rule(self, ruleterm, module):
        if helper.is_term(ruleterm):
            assert isinstance(ruleterm, Callable)
            if ruleterm.signature().eq(predsig):
                return Rule(ruleterm.argument_at(0), ruleterm.argument_at(1), module)
            else:
                return Rule(ruleterm, None, module)
        elif isinstance(ruleterm, Atom):
            return Rule(ruleterm, None, module)
        else:
            error.throw_type_error("callable", ruleterm)

    @jit.elidable_promote('all')
    def get_builtin(self, signature):
        from prolog import builtin # for the side-effects
        return signature.get_extra("builtin")


    # _____________________________________________________
    # parsing-related functionality

    def _build_and_run(self, tree, source_string, file_name, similarity=None):
        assert self is not None # for the annotator (!)
        from prolog.interpreter.parsing import TermBuilder
        builder = TermBuilder()
        term = builder.build_query(tree)
        if isinstance(term, Callable) and term.signature().eq(callsig):
            self.run_query_in_current(term.argument_at(0))
        else:
            term = self._term_expand(term)
            rule = self.add_rule(term)
            if similarity is not None:
                rule.scores = similarity.get_initial_rule_scores(term)
            else:
                rule.scores = [1.0]

            rule.file_name = file_name
            rule._init_source_info(tree, source_string)
            
            if len(rule.source.split(':-')) < 2:
                source = rule.source
                startArgs = source.find('(') + 1
                pred = source[:startArgs-1]
                endArgs = source.find(')') 
                nargs = len(source[startArgs:endArgs].split(','))
                key = pred + '/' + str(nargs)
                if key in self.rule2index:
                    rule_index = self.rule2index[key]
                    self.index2rule[rule_index] = key
            else:
                if rule.source in self.rule2index:
                    rule_index = self.rule2index[rule.source]
                    self.index2rule[rule_index] = rule

    def _term_expand(self, term):
        if self.modulewrapper.system is not None:
            v = BindingVar()
            call = Callable.build("term_expand", [term, v])
            try:
                self.run_query_in_current(call)
            except error.UnificationFailed:
                v = BindingVar()
                call = Callable.build("term_expand", [term, v])
                self.run_query(call, self.modulewrapper.system)
            term = v.dereference(None)
        return term

    def runstring(self, s, file_name=None, similarity=None):
        from prolog.interpreter.parsing import parse_file
        parse_file(s, None, Engine._build_and_run, self, file_name=file_name, similarity=similarity)

    def parse(self, s, file_name=None):
        from prolog.interpreter.parsing import parse_file, TermBuilder
        builder = TermBuilder()
        trees = parse_file(s, None, file_name=file_name)
        terms = builder.build_many(trees)
        return terms, builder.varname_to_var

    def getoperations(self):
        from prolog.interpreter.parsing import default_operations
        if self.operations is None:
            return default_operations
        return self.operations

    # _____________________________________________________
    # Prolog execution


    def run_query(self, query, module, continuation=None, embedding = None):
        assert isinstance(module, Module)
        rule = module._toplevel_rule
        fcont = DoneFailureContinuation(self)
        if continuation is None:
            continuation = CutScopeNotifier(self, DoneSuccessContinuation(self), fcont)
        continuation = BodyContinuation(self, rule, continuation, query)
        continuation.first = True
        return driver(continuation, fcont, Heap(), self)

    def run_query_in_current(self, query, continuation=None):
        module = self.modulewrapper.current_module
        return self.run_query(query, module, continuation)

    
    #EDIT: _rule_to_simchain takes in a rulechain and returns a list containing
    #names of all names with attributed similarity scores including itself
    def _rule_to_simchain(self, rulechain):
        simchain = []
        sig = rulechain.signature
        if (sig.name in self.similarity.simdict):
            simchain = self.similarity.simdict[sig.name]
        return simchain  

    def _print_rulechain(self, rulechain):
        chain = []
        x = rulechain
        while x is not None:
            chain.append(x)
            x = x.next
        print "Rulechain is", chain
        return

    def get_query_string(self, query):
        queryargs = []
        for v in query.arguments():
            if isinstance(v, Atom):
                queryargs.append(v.name())
            elif isinstance(v, BindingVar):
                if (v.binding is not None):
                    try:
                        queryargs.append(v.binding.name())
                    except:
                        queryargs.append(str(v.binding.num))
                else:
                    queryargs.append(v.name())
            elif isinstance(v, Number):
                queryargs.append(str(v.num))

        querystring = query._signature.name + '(' + ','.join(queryargs) + ').'
        return querystring

    # pass in heap
    def get_next_rule(self, rule_dist, heap):  
        # TODO: consider setting a threshold below which rules are not tried

        if (rule_dist is None):
            return None, 0.0, rule_dist

        if isinstance(rule_dist, tuple):
            facts, rule_dist= rule_dist
            if (facts == []):
                return self.get_next_rule(rule_dist, heap)
            else:
                return facts[0], 1.0, (facts[1:], rule_dist)
            
        if (np.amax(rule_dist) == -np.inf):
            return None, 0.0, rule_dist

        rule_index = (-rule_dist).argsort()[0]
        score = rule_dist[rule_index]
        
        # set value of ruleindex in rule_dist to -inf so it isn't selected
        # again for the same query
        rule_dist[rule_index] = -np.inf
        rule = self.index2rule[rule_index]
        #print 'rule output', rule
        #print '\n'

        return rule, score, rule_dist

    def get_top_ten(self, rule_dist):
        if (rule_dist is not None):
            rule_indices = (-rule_dist).argsort()[:10]
            for i in rule_indices: 
                if isFact(self.index2rule[i]):
                    print(self.index2rule[i])
                else:
                    print(self.index2rule[i].source)

   
    def regularcall(self, query, rule, scont, fcont, heap):
        # do a real call
        signature = query.signature()
        module = rule.module
        function = self._get_function(signature, module, query)
        query = function.add_meta_prefixes(query, module.nameatom)
        startrulechain = jit.hint(function.rulechain, promote=True)
        rulechain, simchain = startrulechain.find_applicable_rule(query, heap, self.similarity)
        if rulechain is None:
            raise error.UnificationFailed
        if heap.depth > self.max_depth:
            raise error.UnificationFailed

        scont, fcont, heap = regular_make_rule_conts(self, scont, fcont, heap, query, rulechain)
        return scont, fcont, heap



        
    def call(self, query, rule, scont, fcont, heap, k = 0, parent_rule = None, sister_rule = None, first = False):
        if isinstance(query, Var):
            query = query.dereference(heap)
        if not isinstance(query, Callable):
            if isinstance(query, Var):
                raise error.throw_instantiation_error()
            raise error.throw_type_error('callable', query)

        signature = query.signature()        
        builtin = self.get_builtin(signature)

        rule1 = None
        try: 
            rule1 = scont.rule
        except:
            pass
        log_sister = (rule1 == rule) and scont.from_and 
        
        if builtin is not None:
            if (signature.name != ','):
                if (log_sister):
                    scont.sister_rule = sister_rule
            scont = BuiltinContinuation(self, rule, scont, builtin, query, parent_rule, sister_rule)
            return scont, fcont, heap

        if first:
            return self.regularcall(query, rule, scont, fcont, heap)


        query_string = self.get_query_string(query)
        rule = (parent_rule, sister_rule)
        var_start = query_string.index('(')
        var_end = query_string.index(')')
        var = query_string[var_start+1:var_end]
        var = map(str.strip, var)
        if (len(var) < 4):
            var += [None] * (4-len(var))
        rule_dist, term_dist = run_model(query_string, rule, var)
        
        print('TOP TEN RULES')
        self.get_top_ten(rule_dist)
        print('\n')
        rule, score, rule_dist = self.get_next_rule(rule_dist, heap)

        if (rule is None):
            k = self.maxk

        if (k < self.maxk):
            k = k+1

        if heap.depth > self.max_depth:
            raise error.UnificationFailed

        scont, fcont, heap = _make_rule_conts(self, scont, fcont, heap,\
                              query, rule, k, rule_dist, score, \
                              parent_rule, sister_rule, log_sister)
        return scont, fcont, heap


    def call_in_module(self, query, module, scont, fcont, heap):
        return self.call(query, module._toplevel_rule, scont, fcont, heap)

    def _get_function(self, signature, module, query): 
        function = module.lookup(signature)
        if function.rulechain is None and self.modulewrapper.system is not None:
            function = self.modulewrapper.system.lookup(signature)
        if function.rulechain is None:
            raise error.UnificationFailed
            # TODO Is this okay?
            # return error.throw_existence_error(
            #         "procedure", query.get_prolog_signature())
        return function

    # _____________________________________________________
    # module handling

    def switch_module(self, modulename):
        m = self.modulewrapper
        try:
            m.current_module = m.modules[modulename]
        except KeyError:
            module = Module(modulename)
            m.modules[modulename] = module
            m.current_module = module

    # _____________________________________________________
    # error handling

    @jit.unroll_safe
    def throw(self, exc, scont, fcont, heap, rule_likely_source=None):
        from prolog.interpreter import memo
        exc_term = exc.term
        # copy to make sure that variables in the exception that are
        # backtracked by the revert_upto below have the right value.
        exc_term = exc.term.copy(heap, memo.CopyMemo())
        orig_scont = scont
        while not scont.is_done():
            if not isinstance(scont, CatchingDelimiter):
                scont = scont.nextcont
                continue
            discard_heap = scont.heap
            heap = heap.revert_upto(discard_heap)
            try:
                scont.catcher.unify(exc_term, heap)
            except error.UnificationFailed:
                scont = scont.nextcont
            else:
                return self.call(
                    scont.recover, scont.rule, scont.nextcont, scont.fcont, heap)
        raise error.UncaughtError(exc_term, exc.sig_context, rule_likely_source, orig_scont)

    def __freeze__(self):
        return True
#____________________________________________________________________
# END OF ENGINE

def _log_meta_info(heap, rule, parent_rule, sister_rule, query):
    """if (parent_rule is not None):
        parent_rule = parent_rule.source
    if (sister_rule is not None):
        sister_rule = sister_rule.source
    heap.rules.append((parent_rule, sister_rule, rule.source))
    heap.queries.append(query)"""
    heap.rules.append(rule.source)

def get_rules_from_fact(engine, rule, rule_dist, query, heap, rules = None):
    module = engine.modulewrapper.current_module 
    rule = rule.split('/')
    name = rule[0]
    nargs = int(rule[1])
    sig = Signature.getsignature(name, nargs)
    function = module.lookup(sig)
    simchain = []
    if (function.rulechain is None):
        return None #handle by UnificationFailed
    if rules is None:
        rules = []
    #engine._print_rulechain(function.rulechain)
    #print('\n')
    startrulechain = jit.hint(function.rulechain, promote = True)
    if startrulechain is not None:
        rulechain, simchain = startrulechain.find_applicable_rule(query, heap, engine.similarity, simchain, module, nargs) # TODO similarity
        while rulechain is not None:
            rules.append(jit.hint(rulechain, promote = True)) 
            rulechain, simchain = rulechain.find_next_applicable_rule(query, heap, engine.similarity, simchain, module, nargs)
    
     

    #print(rules)
    return rules


def regular_make_rule_conts(engine, scont, fcont, heap, query, rulechain):
    rule = jit.hint(rulechain, promote=True)
    
    if rule.contains_cut:
        scont = CutScopeNotifier.insert_scope_notifier(
                engine, scont, fcont)
    try:
        shared_env = rule.unify_and_standardize_apart_head(heap, query, similarity=engine.similarity)
    except error.UnificationFailed:
        return fcont.fail(heap)

    scont = RuleContinuation.make(shared_env, engine, scont, rule, query, None, None)


    return scont, fcont, heap

def _make_rule_conts(engine, scont, fcont, heap, query, rule, k, rule_dist, score, parent_rule, sister_rule, log_sister):

    #print "rule", rule
    if (rule is None):
       raise error.UnificationFailed 
    # facts are represented form <predicate>/<arity>
    # need to find an actual fact
    # need to modify this to find multiple facts
    if (isFact(rule)):
        rules = get_rules_from_fact(engine, rule, rule_dist, query, heap)
        k = k - len(rules)

        if (rules == []):
             k += 1
             nextrule, nextscore, rule_dist = engine.get_next_rule(rule_dist, heap)
             if isFact(nextrule):
                return _make_rule_conts(engine, scont, fcont, heap, query, nextrule, k, rule_dist, score, parent_rule, sister_rule, log_sister)
             raise error.UnificationFailed
        else:
            rule = rules[0]
            rule_dist = (rules[1:], rule_dist)
    if rule.contains_cut:
        scont = CutScopeNotifier.insert_scope_notifier(
                engine, scont, fcont)
    
    if k < engine.maxk:
        nextrule, nextscore, rule_dist = engine.get_next_rule(rule_dist, heap)
        k = k+1

        if (nextrule is None):
            k = engine.maxk
        else:
            fcont = UserCallContinuation.make(query.arguments(), engine, \
                    scont, fcont, heap, query, nextrule, k, rule_dist,  \
                    nextscore, parent_rule, sister_rule, log_sister)
            heap = heap.branch()

    try:
        #print "unifying ", rule.source, query
        shared_env = rule.unify_and_standardize_apart_head(heap, query, \
                     similarity=engine.similarity)
        if log_sister:
            scont.sister_rule = rule

    except error.UnificationFailed:
        #print "FAILED"
        return fcont.fail(heap)
    
    scont = RuleContinuation.make(shared_env, engine, scont, rule, query, parent_rule, sister_rule)

    return scont, fcont, heap

# ___________________________________________________________________
# Continuation classes

def _dot(self, seen):
    if self in seen:
        return
    seen.add(self)
    yield '%s [label="%s", shape=box]' % (id(self), repr(self)[:50])
    for key, value in self.__dict__.iteritems():
        if hasattr(value, "_dot"):
            yield "%s -> %s [label=%s]" % (id(self), id(value), key)
            for line in value._dot(seen):
                yield line

class MetaCont(type):
    def __new__(cls, name, bases, dct):
        dct['cont_type_name'] = name
        return type.__new__(cls, name, bases, dct)

class Continuation(object):
    """ Represents a continuation of the Prolog computation. This can be seen
    as an RPython-compatible way to express closures. """

    __metaclass__ = MetaCont

    def __init__(self, engine, nextcont):
        self.engine = engine
        self.nextcont = nextcont

    def activate(self, fcont, heap):
        """ Follow the continuation. heap is the heap that should be used while
        doing so, fcont the failure continuation that should be activated in
        case this continuation fails. This method can only be called once, i.e.
        it can destruct this object. 
        
        The method should return a triple (next cont, failure cont, heap)"""
        raise NotImplementedError("abstract base class")

    def is_done(self):
        return False

    def find_end_of_cut(self):
        return self.nextcont.find_end_of_cut()

    _dot = _dot

class ContinuationWithRule(Continuation):
    """ This class represents continuations which need
    to have a reference to the current rule
    (e.g. to get at the module). """

    def __init__(self, engine, nextcont, rule):
        Continuation.__init__(self, engine, nextcont)
        assert isinstance(rule, Rule)
        self.rule = rule

def view(*objects, **names):
    from dotviewer import graphclient
    content = ["digraph G{"]
    seen = set()
    for obj in list(objects) + names.values():
        content.extend(obj._dot(seen))
    for key, value in names.items():
        content.append("%s -> %s" % (key, id(value)))
    content.append("}")
    p = py.test.ensuretemp("prolog").join("temp.dot")
    p.write("\n".join(content))
    graphclient.display_dot_file(str(p))


class FailureContinuation(object):
    """ A continuation that can represent failures. It has a .fail method that
    is called to figure out what should happen on a failure.
    """
    def __init__(self, engine, nextcont, orig_fcont, heap):
        self.engine = engine
        self.nextcont = nextcont
        self.orig_fcont = orig_fcont
        self.undoheap = heap

    def fail(self, heap):
        """ Needs to be called to get the new success continuation.
        Returns a tuple (next cont, failure cont, heap)
        """
        raise NotImplementedError("abstract base class")

    def cut(self, upto, heap):
        """ Cut away choice points till upto. """
        if self is upto:
            return
        heap = self.undoheap.discard(heap)
        self.orig_fcont.cut(upto, heap)

    def is_done(self):
        return False

    _dot = _dot

def make_failure_continuation(make_func):
    class C(FailureContinuation):
        def __init__(self, engine, scont, fcont, heap, *state):
            FailureContinuation.__init__(self, engine, scont, fcont, heap)
            self.state = state

        def fail(self, heap):
            heap = heap.revert_upto(self.undoheap, discard_choicepoint=True)
            return make_func(C, self.engine, self.nextcont, self.orig_fcont,
                             heap, *self.state)
    C.__name__ = make_func.__name__ + "FailureContinuation"
    def make_func_wrapper(*args):
        return make_func(C, *args)
    make_func_wrapper.__name__ = make_func.__name__ + "_wrapper"
    return make_func_wrapper

class DoneSuccessContinuation(Continuation):
    def __init__(self, engine):
        Continuation.__init__(self, engine, None)
        self.failed = False

    def is_done(self):
        return True

class DoneFailureContinuation(FailureContinuation):
    def __init__(self, engine):
        FailureContinuation.__init__(self, engine, None, None, None)

    def fail(self, heap):
        scont = DoneSuccessContinuation(self.engine)
        scont.failed = True
        return scont, self, heap

    def is_done(self):
        return True


class BodyContinuation(ContinuationWithRule):
    """ Represents a bit of Prolog code that is still to be called. """
    def __init__(self, engine, rule, nextcont, body):
        ContinuationWithRule.__init__(self, engine, nextcont, rule)
        self.body = body
        self.parent_rule = None
        self.sister_rule = None
        self.from_and = False
        self.first = False

    def activate(self, fcont, heap):
        return self.engine.call(self.body, self.rule, self.nextcont, fcont, heap, 0, self.parent_rule, self.sister_rule, first = self.first)

    def __repr__(self):
        return "<BodyContinuation %r>" % (self.body, )

class BuiltinContinuation(ContinuationWithRule):
    """ Represents the call to a builtin. """
    def __init__(self, engine, rule, nextcont, builtin, query, parent_rule, sister_rule):
        ContinuationWithRule.__init__(self, engine, nextcont, rule)
        self.builtin = builtin
        self.query = query
        self.sister_rule = sister_rule
        
    def activate(self, fcont, heap):
        return self.builtin.call(self.engine, self.query, self.rule,
                self.nextcont, fcont, heap, self.sister_rule)

    def __repr__(self):
        return "<BuiltinContinuation %r, %r>" % (self.builtin, self.query, )


@inline_small_list(immutable=True)
class UserCallContinuation(FailureContinuation):
    def __init__(self, engine, nextcont, orig_fcont, heap, query, rule, k, rule_dist, score, parent_rule, sister_rule, log_sister):
        FailureContinuation.__init__(self, engine, nextcont, orig_fcont, heap)
        self.query = query
        self.rule = rule
        self.k = k
        self.score = score
        self.rule_dist = rule_dist

        # fields to get parent and sister rules
        self.parent_rule = parent_rule
        self.sister_rule = sister_rule
        self.log_sister = log_sister

    def fail(self, heap):
        heap = heap.revert_upto(self.undoheap, discard_choicepoint=True)
        query = self.query
        rule = self.rule
        k = self.k
        score = self.score
        rule_dist = self.rule_dist
        parent_rule = self.parent_rule
        sister_rule = self.sister_rule
        log_sister = self.log_sister
        return _make_rule_conts(self.engine, self.nextcont, self.orig_fcont,
                                heap, query, rule, k, rule_dist, score, 
                                parent_rule, sister_rule, log_sister)


    def __repr__(self):
        query = self.rulechain.build_query(self._get_full_list())
        return "<UserCallContinuation query=%r rule=%r>" % (
                query, self.rulechain)

@inline_small_list(immutable=True)
# pass in query here and also log query
class RuleContinuation(ContinuationWithRule):
    """ A Continuation that represents the application of a rule, i.e.:
        - standardizing apart of the rule
        - unifying the rule head with the query
        - calling the body of the rule
    """

    def __init__(self, engine, nextcont, rule, query, parent_rule, sister_rule):
        ContinuationWithRule.__init__(self, engine, nextcont, rule)
        self.query = engine.get_query_string(query)
        self.parent_rule = parent_rule
        self.sister_rule = sister_rule
    
    def activate(self, fcont, heap):
        nextcont = self.nextcont
        rule = jit.promote(self.rule)
        nextcall = rule.clone_body_from_rulecont(heap, self)
        _log_meta_info(heap, rule, self.parent_rule, self.sister_rule, self.query)
        parent_rule = rule
        if nextcall is not None:
            heap.depth += 1
            return self.engine.call(nextcall, self.rule, nextcont, fcont, heap, 0, parent_rule, None)
        else:
            cont = nextcont
        return cont, fcont, heap

    def __repr__(self):
        return "<RuleContinuation rule=%r>" % (self.rule)

class CutScopeNotifier(Continuation):
    def __init__(self, engine, nextcont, fcont_after_cut):
        Continuation.__init__(self, engine, nextcont)
        self.fcont_after_cut = fcont_after_cut

    @staticmethod
    def insert_scope_notifier(engine, nextcont, fcont):
        if isinstance(nextcont, CutScopeNotifier) and nextcont.fcont_after_cut is fcont:
            return nextcont
        return CutScopeNotifier(engine, nextcont, fcont)

    def find_end_of_cut(self):
        return self.fcont_after_cut

    def activate(self, fcont, heap):
        return self.nextcont, fcont, heap


class CatchingDelimiter(ContinuationWithRule):
    def __init__(self, engine, rule, nextcont, fcont, catcher, recover, heap):
        ContinuationWithRule.__init__(self, engine, nextcont, rule)
        self.catcher = catcher
        self.recover = recover
        self.fcont = fcont
        self.heap = heap

    def activate(self, fcont, heap):
        return self.nextcont, fcont, heap

    def __repr__(self):
        return "<CatchingDelimiter catcher=%s recover=%s>" % (self.catcher, self.recover)
