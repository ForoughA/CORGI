
import py
from prolog.interpreter.parsing import TermBuilder
from prolog.interpreter.parsing import parse_query_term, get_engine
from prolog.interpreter.error import UnificationFailed
from prolog.interpreter.continuation import Heap, Engine
from prolog.interpreter import error
from prolog.interpreter.test.tool import collect_all, assert_false, assert_true
from prolog.interpreter.test.tool import prolog_raises
from prolog.interpreter.similarity import Similarity

def test_or():
    similarity = Similarity(0.1)
    similarity.set_score('believes_in', 'asserts', 0.2)
    similarity.set_score('free_will', 'free_goodies', 0.8)
    e = get_engine("""
        believes_in(alice, free_will).
       asserts(X,Y) :- believes_in(X, Y).
    """,
                   similarity=similarity
                   )
    # result = assert_true("believes_in(X, free_will).", e)
    # print result
    result = collect_all(e, "asserts(alice, free_goodies).")
    print result


if __name__ == '__main__':
    test_or()
