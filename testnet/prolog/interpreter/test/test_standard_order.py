import py
from prolog.interpreter.test.tool import assert_false, assert_true
from prolog.interpreter.continuation import Engine

e = Engine(load_system=True)

# ---[ Coarse type-based comparisons ]---

# Var < Float = Number = BigInt < Atom < Term
# Note that strings are not yet implemented.

def test_basic_order():
    assert_true("X @< 1.0.", e)
    assert_true("666.0 @< a.", e)
    assert_true("1024 @< a.", e)
    assert_true("%s @< a." % (2**64*6, ), e)
    assert_true("a @< c(x, y).", e)

    assert_false("a @> c(x, y).", e)
    assert_false("%s @> a." % (2**64*6, ), e)
    assert_false("1024 @> a.", e)
    assert_false("666.0 @> a.", e)
    assert_false("X @> 1.0.", e)

    assert_false("a @>= c(x, y).", e)
    assert_false("%s @>= a." % (2**64*6, ), e)
    assert_false("1024 @>= a.", e)
    assert_false("666.0 @>= a.", e)
    assert_false("X @>= 1.0.", e)

    assert_true("1.0 @> X.", e)
    assert_true("a @> 666.0.", e)
    assert_true("a @> 1024.", e)
    assert_true("a @> %s." % (2**64*6, ), e)
    assert_true("c(x, y) @> a.", e)

# ---[ Inter-numeric comparisons ]---

def test_number_vs_number_order():
    assert_true("1 =@= 1.", e)

    assert_false("1 @< 1.", e)
    assert_false("1 @> 1.", e)
    assert_true("1 @< 987.", e)
    assert_true("7877 @> 7876.", e)

    assert_true("1 @=< 1.", e)
    assert_true("1 @>= 1.", e)
    assert_false("1 @=< 0.", e)
    assert_false("0 @>= 1.", e)

def test_bigint_vs_bigint_order():
    big = 2**64*4

    assert_true("%s =@= %s." % (big, big), e)

    assert_false("%s @> %s." % (big, big), e)
    assert_false("%s @< %s." % (big, big), e)
    assert_true("%s @> %s." % (big + 1, big), e)
    assert_true("%s @< %s." % (big, big + 1), e)

    assert_true("%s @>= %s." % (big, big), e)
    assert_true("%s @=< %s." % (big, big), e)
    assert_false("%s @>= %s." % (big, big+1), e)
    assert_false("%s @=< %s." % (big+1, big), e)

def test_float_vs_float_order():
    assert_true("1.0 =@= 1.0.", e)

    assert_false("1.0 @< 1.0.", e)
    assert_false("1.0 @> 1.0.", e)
    assert_true("1.0 @< 987.0.", e)
    assert_true("7877.0 @> 7876.0.", e)

    assert_true("1.0 @=< 1.0.", e)
    assert_true("1.0 @>= 1.0.", e)
    assert_false("1.1 @=< 1.0.", e)
    assert_false("0.95 @>= 1.0.", e)

def test_float_vs_number_order():
    assert_true("66.0 =@= 66.", e)

    assert_false("66.0 @< 66.", e)
    assert_false("66.0 @> 66.", e)
    assert_true("1.0 @< 2.", e)
    assert_true("1.0 @> -55.", e)

    assert_true("66.0 @=< 66.", e)
    assert_true("66.0 @>= 66.", e)
    assert_false("66.01 @=< 66.", e)
    assert_false("65.499 @>= 66.", e)

def test_number_vs_float_order():
    assert_true("66 =@= 66.0.", e)

    assert_false("66 @< 66.0.", e)
    assert_false("66 @> 66.0.", e)
    assert_true("66 @< 690.", e)
    assert_true("666 @> 555.", e)

    assert_true("66 @=< 66.0.", e)
    assert_true("66 @>= 66.0.", e)
    assert_false("66 @=< 58.1.", e)
    assert_false("66 @>= 67.6666.", e)

def test_number_vs_bigint_order():
    big = 2**64*4
    assert_true("1 @< %s." % big, e)
    assert_false("1 @> %s." % big, e)

def test_bigint_vs_number_order():
    big = 2**64*4
    assert_true("%s @> 1." % big, e)
    assert_false("%s @< 1." % big, e)

def test_bigint_vs_float_order():
    big = 2**64*4
    assert_false("%s @< 1.0." % big, e)
    assert_true("%s @> 1.0." % big, e)

def test_float_vs_bigint_order():
    big = 2**64*4
    assert_true("1.0 @< %s." % big, e)
    assert_false("1.0 @> %s." % big, e)

# ---[ Inter-var comparisons ]---

# Really hard to test, as hte variable binding can change the address.
# e.g. The following may pass or fail...

#def test_var_vs_var_order():
#    query = """
#            X @=< TREE, X @=< MARSHMALLOW, X @=< COMPACT_DISC,
#            TREE @>= X,  MARSHMALLOW @>= X, COMPACT_DISC @>= X,
#            (X =@= TREE; X =@= MARSHMALLOW; X =@= COMPACT_DISC).
#            """
#    assert_true(query, e)

# ---[ Inter-atom comparisons ]---

def test_atom_vs_atom_order():
    assert_true("a =@= a.", e)

    assert_true("a @< b.", e)
    assert_true("a @=< b.", e)
    assert_true("a @=< a.", e)

    assert_true("z @> reallylong.", e)
    assert_true("z @>= reallylong.", e)
    assert_true("zzz @>= zzz.", e)

# ---[ Inter-term comparisons ]---

def test_term_vs_term_order():
    assert_true("f(a, b, c) =@= f(a, b, c).", e)
    assert_true("f(a, b, c) @=< f(a, b, c).", e)
    assert_true("f(a, b, c) @>= f(a, b, c).", e)

    assert_true("f(a, b) @< f(a, b, c).", e)
    assert_true("f(a, b, c) @> f(a, b).", e)

    assert_true("f(g(1), g(2), g(3)) =@= f(g(1), g(2), g(3)).", e)
    assert_true("f(g(1), g(2), g(3)) @>= f(g(1), g(2), g(3)).", e)
    assert_true("f(g(1), g(2), g(3)) @=< f(g(1), g(2), g(3)).", e)

    assert_true("f(g(1), g(2), g(3)) @> f(g(1), g(2)).", e)
    assert_true("f(g(1), g(2)) @< f(g(1), g(2), g(3)).", e)

    assert_true("f(g(1), g(2), g(3)) @> f(g(1), g(2), g(2)).", e)
    assert_true("f(g(1), g(2), g(3)) @> f(g(1), g(2), f(3)).", e)

    assert_true("ff(1) @< fff(1).", e)
    assert_true("ff(1) @=< fff(1).", e)
    assert_true("fff(1) @> ff(1).", e)
    assert_true("fff(1) @>= ff(1).", e)
