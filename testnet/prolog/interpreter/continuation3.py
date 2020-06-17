import py
import time
from rpython.rlib import jit
from rpython.rlib.objectmodel import we_are_translated, specialize
from prolog.interpreter import error
from prolog.interpreter import helper
from prolog.interpreter.term import Term, Atom, BindingVar, Callable, Var
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
    def __init__(self, load_system=False, similarity=None, max_depth=10, lambda_cut=0.1):
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
            rule.file_name = file_name
            rule._init_source_info(tree, source_string)

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

    def run_query(self, query, module, continuation=None):
        assert isinstance(module, Module)
        rule = module._toplevel_rule
        fcont = DoneFailureContinuation(self)
        if continuation is None:
            continuation = CutScopeNotifier(self, DoneSuccessContinuation(self), fcont)
        continuation = BodyContinuation(self, rule, continuation, query)
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
    
    def call(self, query, rule, scont, fcont, heap):
        if isinstance(query, Var):
            query = query.dereference(heap)
        if not isinstance(query, Callable):
            if isinstance(query, Var):
                raise error.throw_instantiation_error()
            raise error.throw_type_error('callable', query)
        signature = query.signature()        
        builtin = self.get_builtin(signature)
        if builtin is not None:
            #print(signature)
            return BuiltinContinuation(self, rule, scont, builtin, query), fcont, heap

        # do a real call
        module = rule.module
        function = self._get_function(signature, module, query)
        query = function.add_meta_prefixes(query, module.nameatom) #makes a Callable?
        # _print_rulechain(function.rulechain)
        simchain = self._rule_to_simchain(function.rulechain)
        #print "Simchain is", simchain 
        startrulechain = jit.hint(function.rulechain, promote=True)
        rulechain, simchain = startrulechain.find_applicable_rule(query, heap, self.similarity, simchain, module, len(startrulechain.headargs))
        #print "rule: ", rulechain
        #print "newsimchain: ", simchain        
        if rulechain is None:
            raise error.UnificationFailed
        if heap.depth > self.max_depth:
            raise error.UnificationFailed

        scont, fcont, heap = _make_rule_conts(self, scont, fcont, heap, query, rulechain, simchain)
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


def _log_meta_info(heap, rule, query):
    heap.rules.append(rule.source)

def _make_rule_conts(engine, scont, fcont, heap, query, rulechain, simchain):
    rule = jit.hint(rulechain, promote=True)
    if rule.contains_cut:
        scont = CutScopeNotifier.insert_scope_notifier(
                engine, scont, fcont)
    
    restchain, simchain = rule.find_next_applicable_rule(query, heap=heap, similarity=engine.similarity, simchain = simchain, module = rule.module, nargs = len(rule.headargs))
    
    #print "nextrule", restchain
    #print "nextsimchain", simchain

    if restchain is not None:
        fcont = UserCallContinuation.make(query.arguments(), engine, scont, fcont, heap, restchain, simchain, query)
        heap = heap.branch()

    try:
        # TODO: do the query logging here??
        queryargs = []
        for v in query.arguments():
            if isinstance(v, Atom):
                queryargs.append(v.name())
            if isinstance(v, BindingVar):
                if (v.binding is not None):
                    queryargs.append(v.binding.name())
                else:
                    queryargs.append(v.name())
                        
            
        querystring = query._signature.name + '(' + ','.join(queryargs) + ').'
        heap.queries.append(querystring)
        shared_env = rule.unify_and_standardize_apart_head(heap, query, similarity=engine.similarity)
    
    except error.UnificationFailed:
        #unlog queries
        heap.queries = heap.queries[:-1]
        return fcont.fail(heap)


    scont = RuleContinuation.make(shared_env, engine, scont, rule, query)

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

    def activate(self, fcont, heap):
        return self.engine.call(self.body, self.rule, self.nextcont, fcont, heap)

    def __repr__(self):
        return "<BodyContinuation %r>" % (self.body, )

class BuiltinContinuation(ContinuationWithRule):
    """ Represents the call to a builtin. """
    def __init__(self, engine, rule, nextcont, builtin, query):
        ContinuationWithRule.__init__(self, engine, nextcont, rule)
        self.builtin = builtin
        self.query = query

    def activate(self, fcont, heap):
        return self.builtin.call(self.engine, self.query, self.rule,
                self.nextcont, fcont, heap)

    def __repr__(self):
        return "<BuiltinContinuation %r, %r>" % (self.builtin, self.query, )


@inline_small_list(immutable=True)
class UserCallContinuation(FailureContinuation):
    def __init__(self, engine, nextcont, orig_fcont, heap, rulechain, simchain, query):
        FailureContinuation.__init__(self, engine, nextcont, orig_fcont, heap)
        self.rulechain = rulechain
        self.simchain = simchain
        self.query = query

    def fail(self, heap):
        heap = heap.revert_upto(self.undoheap, discard_choicepoint=True)
        query = self.query
        #query = self.rulechain.build_query(self._get_full_list())
        return _make_rule_conts(self.engine, self.nextcont, self.orig_fcont,
                                heap, query, self.rulechain, self.simchain)


    def __repr__(self):
        query = self.rulechain.build_query(self._get_full_list())
        return "<UserCallContinuation query=%r rule=%r>" % (
                query, self.rulechain)

@inline_small_list(immutable=True)
class RuleContinuation(ContinuationWithRule):
    """ A Continuation that represents the application of a rule, i.e.:
        - standardizing apart of the rule
        - unifying the rule head with the query
        - calling the body of the rule
    """

    def __init__(self, engine, nextcont, rule, query):
        ContinuationWithRule.__init__(self, engine, nextcont, rule)
        self.query = query
    def activate(self, fcont, heap):
        nextcont = self.nextcont
        rule = jit.promote(self.rule)
        _log_meta_info(heap, rule, self.query)
        nextcall = rule.clone_body_from_rulecont(heap, self)
        if nextcall is not None:
            heap.depth += 1
            return self.engine.call(nextcall, self.rule, nextcont, fcont, heap)
        else:
            cont = nextcont
        return cont, fcont, heap

    def __repr__(self):
        return "<RuleContinuation rule=%r query=%r>" % (self.rule, self.query)

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
