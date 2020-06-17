import sys
from prolog.interpreter.translatedmain import repl, execute

# __________  Entry point  __________

from prolog.interpreter.continuation import Engine, jitdriver
from prolog.interpreter import term
from prolog.interpreter import arithmetic # for side effects
from prolog import builtin # for side effects
from prolog.interpreter.similarity import from_file
#from prolog.interpreter.test.tool import collect_all

from rpython.rlib import jit


def entry_point(argv):
    if len(argv) != 4:
        print "Wrong number of arguments"
        return 1

    with open(argv[1]) as f:
        program = f.read()
    similarity = from_file(argv[2])
    engine = Engine()
    term.DEBUG = False
    # results = collect_all(e, argv[3])
    max_score = 0.0
    # for result in results:
    #     max_score = max(max_score, result[1])
    # print max_score
    return 0

# _____ Define and setup target ___


def target(driver, args):
    driver.exe_name = 'pyrolog-%(backend)s'
    return entry_point, None

def portal(driver):
    from prolog.interpreter.portal import get_portal
    return get_portal(driver)

def jitpolicy(self):
    from rpython.jit.codewriter.policy import JitPolicy
    return JitPolicy()


if __name__ == '__main__':
    entry_point(sys.argv)
