import py
from prolog.interpreter import helper, term, error, continuation, memo
from prolog.builtin.register import expose_builtin

# ___________________________________________________________________
# finding all solutions to a goal

class FindallContinuation(continuation.Continuation):
    def __init__(self, engine, template, heap, scont):
        # nextcont still needs to be set, for correct exception propagation
        continuation.Continuation.__init__(self, engine, scont)
        self.result = []
        self.template = template
        self.heap = heap

    def activate(self, fcont, _):
        m = memo.CopyMemo()
        clone = self.template.copy(self.heap, m)
        self.result.append(clone)
        raise error.UnificationFailed()

class DoneWithFindallContinuation(continuation.FailureContinuation):
    def __init__(self, engine, scont, fcont, heap, collector, bag):
        continuation.FailureContinuation.__init__(self, engine, scont, fcont, heap)
        self.collector = collector
        self.bag = bag

    def fail(self, heap):
        heap = heap.revert_upto(self.undoheap)
        self.bag.unify(helper.wrap_list(self.collector.result), heap)
        return self.nextcont, self.orig_fcont, heap



@expose_builtin("findall", unwrap_spec=['raw', 'callable', 'raw'],
                handles_continuation=True, needs_rule=True)
def impl_findall(engine, heap, rule, template, goal, bag, scont, fcont):
    newheap = heap.branch()
    collector = FindallContinuation(engine, template, heap, scont)
    newscont = continuation.BodyContinuation(engine, rule, collector, goal)
    fcont = DoneWithFindallContinuation(engine, scont, fcont, heap, collector, bag)
    return newscont, fcont, newheap
