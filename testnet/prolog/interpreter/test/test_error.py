import py, pytest

from prolog.interpreter.parsing import get_engine
from prolog.interpreter.parsing import get_query_and_vars
from prolog.interpreter.error import UncaughtError
from prolog.interpreter.signature import Signature

def get_uncaught_error(query, e):
    if isinstance(query, str):
        (query, _) = get_query_and_vars(query)
    return pytest.raises(UncaughtError, e.run_query_in_current, query).value


def test_errstr():
    e = get_engine("""
        f(X) :- drumandbass(X).
    """)
    error = get_uncaught_error("f(X).", e)
    assert error.get_errstr(e) == "Undefined procedure: drumandbass/1"

def test_errstr_user():
    e = get_engine("""
        f(X) :- throw(foo).
    """)
    error = get_uncaught_error("f(X).", e)
    assert error.get_errstr(e) == "Unhandled exception: foo"

def test_exception_knows_rule():
    e = get_engine("""
        f(1).
        f(X) :- drumandbass(X).
    """)
    (t, vs) = get_query_and_vars("f(X), X = 2.")

    m = e.modulewrapper
    sig = t.argument_at(0).signature()
    rule = m.user_module.lookup(sig).rulechain.next
    error = get_uncaught_error(t, e)
    assert error.rule is rule

def test_exception_knows_rule_toplevel():
    # toplevel rule
    e = get_engine("")
    m = e.modulewrapper
    error = get_uncaught_error("drumandbass(X).", e)
    assert error.rule is m.current_module._toplevel_rule

def test_exception_knows_rule_change_back_to_earlier_rule():
    e = get_engine("""
        g(a).
        f(X) :- g(X), drumandbass(X).
    """)
    (t, vs) = get_query_and_vars("f(X).")

    m = e.modulewrapper
    sig = t.signature()
    rule = m.user_module.lookup(sig).rulechain

    error = get_uncaught_error(t, e)
    assert error.rule is rule

def test_exception_knows_builtin_signature():
    e = get_engine("""
        f(X, Y) :- atom_length(X, Y).
    """)
    error = get_uncaught_error("f(1, Y).", e)
    assert error.sig_context == Signature.getsignature("atom_length", 2)

def test_traceback():
    e = get_engine("""
        h(y).
        g(a).
        g(_) :- throw(foo).
        f(X, Y) :- g(X), h(Y).
    """)
    error = get_uncaught_error("f(1, Y).", e)
    sig_g = Signature.getsignature("g", 1)
    sig_f = Signature.getsignature("f", 2)
    m = e.modulewrapper
    rule_f = m.user_module.lookup(sig_f).rulechain
    rule_g = m.user_module.lookup(sig_g).rulechain.next
    tb = error.traceback
    assert tb.rule is rule_f
    assert tb.next.rule is rule_g
    assert tb.next.next is None

@pytest.mark.xfail
def test_traceback_in_if():
    e = get_engine("""
        h(y).
        g(a).
        g(_) :- throw(foo).
        f(X, Y) :- (g(X) -> X = 1 ; X = 2), h(Y).
    """)
    error = get_uncaught_error("f(1, Y).", e)
    sig_g = Signature.getsignature("g", 1)
    sig_f = Signature.getsignature("f", 2)
    m = e.modulewrapper
    rule_f = m.user_module.lookup(sig_f).rulechain
    rule_g = m.user_module.lookup(sig_g).rulechain.next
    tb = error.traceback
    assert tb.rule is rule_f
    assert tb.next.rule is rule_g

def test_traceback_print():
    e = get_engine("""
h(y).
g(a).
g(_) :- throw(foo).
f(X, Y) :-
    g(X),
    h(Y).
:- assert((h :- g(b), true)).
    """)
    error = get_uncaught_error("f(1, Y).", e)
    s = error.format_traceback(e)
    assert s == """\
Traceback (most recent call last):
  File "<unknown>" lines 5-7 in user:f/2
    f(X, Y) :-
        g(X),
        h(Y).
  File "<unknown>" line 4 in user:g/1
    g(_) :- throw(foo).
Unhandled exception: foo"""

    error = get_uncaught_error("h.", e)
    s = error.format_traceback(e)
    assert s == """\
Traceback (most recent call last):
  File "<unknown>" in user:h/0
  File "<unknown>" line 4 in user:g/1
    g(_) :- throw(foo).
Unhandled exception: foo"""

def test_traceback_print_builtin():
    e = get_engine("""
h(y).
g(a).
g(_) :- _ is _.
f(X, Y) :-
    g(X),
    h(Y).
:- assert((h :- g(b), true)).
    """)
    error = get_uncaught_error("f(1, Y).", e)
    s = error.format_traceback(e)
    assert s == """\
Traceback (most recent call last):
  File "<unknown>" lines 5-7 in user:f/2
    f(X, Y) :-
        g(X),
        h(Y).
  File "<unknown>" line 4 in user:g/1
    g(_) :- _ is _.
is/2: arguments not sufficiently instantiated"""

def test_traceback_print_no_context():
    e = get_engine("")
    error = get_uncaught_error("f(1, Y).", e)
    s = error.format_traceback(e)
    assert s == """\
Traceback (most recent call last):
  File "<unknown>" in user:<user toplevel>/0
Undefined procedure: f/2"""
