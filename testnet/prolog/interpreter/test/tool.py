import py
import os
from prolog.interpreter.error import UnificationFailed
from prolog.interpreter.parsing import parse_query_term, get_engine
from prolog.interpreter.continuation import Continuation, Heap, Engine
from prolog.interpreter.parsing import parse_file, TermBuilder

def assert_true(query, e=None):
    if e is None:
        e = Engine()
    terms, vars = e.parse(query)
    term, = terms
    e.run_query_in_current(term)
    return dict([(name, var.dereference(None))
                     for name, var in vars.iteritems()])

def assert_false(query, e=None):
    if e is None:
        e = Engine()
    term = e.parse(query)[0][0]
    py.test.raises(UnificationFailed, e.run_query_in_current, term)

def prolog_raises(exc, query, e=None):
    prolog_catch = "catch(((%s), fail), error(%s), true)." % (query, exc)
    return assert_true(prolog_catch, e)

class CollectAllContinuation(Continuation):
    nextcont = None
    def __init__(self, module, vars, rules, scores, unifications, queries):
        self.heaps = []
        self.vars = vars
        self._candiscard = True
        self.module = module
        # EDIT to get proof trace
        self.rules = rules
        self.unifications = unifications 
        self.scores = scores
        self.queries = queries


    def activate(self, fcont, heap):
        bindings = dict([(name, var.dereference(heap)) for name, var in self.vars.iteritems()])
        self.heaps.append((bindings, heap.score()))
        self.rules.append(heap.rules)
        self.scores.append((heap.predicate_score, heap.entity_score, heap.score()))
        self.unifications.append(heap.unifications)
        self.queries.append(heap.queries)
        #print "restarting computation"
        #if (len(self.heaps) < 3):
        #if (not fcont.is_done()):
        raise UnificationFailed
        #else:
         #   return None, None, None

def collect_all(engine, s, rules, scores, unifications, queries):
    terms, vars = engine.parse(s)
    term, = terms
    collector = CollectAllContinuation(engine.modulewrapper.user_module, vars, rules, scores, unifications, queries)
    py.test.raises(UnificationFailed, engine.run_query, term,
            engine.modulewrapper.current_module, collector)
    #engine.run_query(term, engine.modulewrapper.current_module, collector)
    return collector.heaps

def parse(inp):
    t = parse_file(inp)
    builder = TermBuilder()
    return builder.build(t)

def create_file(name, content):
    with open(name, "w") as f:
        f.write(content)

def delete_file(name):
    os.unlink(name)

def create_dir(name):
    os.mkdir(name)

def delete_dir(name):
    current_dir = os.path.abspath(name)
    items = os.listdir(current_dir)
    for item in items:
        abspath = current_dir + "/" + item
        if os.path.isfile(abspath):
            delete_file(abspath)
        else:
            delete_dir(abspath)
    os.rmdir(current_dir)

def file_content(src):
    with open(src) as f:
        return f.read()

