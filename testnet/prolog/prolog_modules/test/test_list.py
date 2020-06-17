from prolog.interpreter.continuation import Engine
from prolog.interpreter.test.tool import collect_all, assert_false, assert_true

e = Engine(load_system=True)
def test_member():
    assert_true("member(1, [1,2,3]).", e)

def test_not_member():
    assert_false("member(666, [661,662,667,689]).", e)

def test_not_member_of_empty():
    assert_false("member(666, []).", e)

def test_all_members():
    data = [ 2**x for x in range(30) ]
    heaps = collect_all(e, "member(X, %s)." % (str(data)))
    nums = [ h["X"].num for h in heaps ]
    assert nums == data

def test_select():
    assert_true("select(1, [1, 2, 3, 1], [2, 3, 1]).", e)
    assert_true("select(1, [1, 2, 3, 1], [1, 2, 3]).", e)
    assert_false("select(1, [1, 2, 3, 1], [1, 2, 3, 1]).", e)
    assert_false("select(1, [1, 2, 3, 1], [2, 3]).", e)
    assert_false("select(2, [], []).", e)
    assert_false("select(2, [], [X]).", e)

def test_nextto():
    assert_true("nextto(666, 1024, [1, 2, 3, 4, 666, 1024, 8, 8, 8]).", e)
    assert_false("nextto(8, 4, [1, 2, 3, 4, 666, 1024, 8, 8, 8]).", e)
    assert_false("nextto(1, 2, [2, 1]).", e)

    heaps = collect_all(e, "nextto(A, B, [1, 2, 3]).")
    assert(len(heaps) == 2)

def test_memberchk():
    assert_true("memberchk(432, [1, 2, 432, 432, 1]).", e)
    assert_false("memberchk(0, [1, 2, 432, 432, 1]).", e)

def test_subtract():
    assert_true("subtract([1, 2, 3, 4], [2], [1, 3, 4]).", e)
    assert_true("subtract([a, c, d], [b], [a, c, d]).", e)
    assert_true("subtract([a, b, c], [], [a, b, c]).", e)
    assert_true("subtract([1, 1, 6], [1, 6], []).", e)

# This is really hard to test as variable addresses can change as their
# binding changes. This is really a problem with @<. See the test for
# this for a counter-example (../interpreter/test/test_standard_order.py).
#
#def test_min_member_var():
#    assert_true("min_member(X, [TEA, UNIVERSE, SHIRT]), " + \
#            "X @=< TEA, X @=< UNIVERSE, X @=< SHIRT.", e)

def test_min_member_number():
    assert_true("min_member(444, [444,445,999]).", e)

def test_min_member_atom():
    assert_true("min_member(fox, [kamikaze,pebble,fox]).", e)

def test_min_member_string():
    assert_true('min_member("fox", ["pebble","fox","foxy","zzz"]).', e)

def test_min_member_term():
    assert_true('min_member(f(x, 3), [g("0", "1", "999"), a(-9, -9, -9), f(x, 3), f(x, 4)]).', e)

def test_min_member_cmp_var_num():
    assert_true('min_member(X, [-999, 10, 20, X]).', e)

def test_min_member_cmp_num_atom():
    assert_true('min_member(1, [a,b,c,1]).', e)

def test_min_member_cmp_atom_string():
    assert_true('min_member(z, ["yeehaw", "blob", z]).', e)

def test_min_member_cmp_string_compund():
    assert_true('min_member("yeehaw", ["yeehaw", flibble(x, b, b), flibble(y, z)]).', e)

def test_min_member_empty():
    assert_false('min_member(X, []).', e)

# Same issue as test_min_member_var. See above.
#def test_max_member_var():
#    assert_true("max_member(X, [TEA,UNIVERSE, SHIRT]), " + \
#            "X @>= TEA, X @>= UNIVERSE, X @>= SHIRT.", e)

def test_max_member_number():
    assert_true("max_member(999, [444,445,999]).", e)

def test_max_member_atom():
    assert_true("max_member(pebble, [kamikaze,pebble,fox]).", e)

def test_max_member_string():
    assert_true('max_member("zzz", ["pebble","fox","foxy","zzz"]).', e)

def test_max_member_term():
    assert_true('max_member(g("0", "1", "999"), [g("0", "1", a), g("0", "1", "999"), a(-9, -9, -9)]).', e)

def test_max_member_cmp_var_num():
    assert_true('max_member(20, [-999, 10, 20, X]).', e)

def test_max_member_cmp_num_atom():
    assert_true('max_member(c, [a,b,c,1]).', e)

def test_max_member_cmp_atom_string():
    assert_true('max_member("yeehaw", ["yeehaw", "blob", z]).', e)

def test_max_member_cmp_string_compund():
    assert_true('max_member(flibble(x, b, b), ["yeehaw", flibble(x, b, b), flibble(y, z)]).', e)

def test_max_member_empty():
    assert_false('max_member(X, []).', e)

def test_delete():
    assert_true('delete([1, 3, 9, 3, 1, 2, 9], 9, [1, 3, 3, 1, 2]).', e)

def test_length():
    assert_true('length([1, a, f(h)], 3).', e)
    assert_true('length([], 0).', e)
    assert_true('length(List, 6), is_list(List), length(List, 6).', e)
    assert_false('length(X, -1).', e)
    assert_false('length(a, Y).', e)

def test_last():
    assert_true('last([1,2,3], 3).', e)
    assert_true('last([666], 666).', e)
    assert_false('last([], X).', e)
