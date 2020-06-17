import os, sys
from rpython.rlib.parsing.parsing import ParseError
from rpython.rlib.parsing.deterministic import LexerError
from prolog.interpreter.parsing import get_query_and_vars
from prolog.interpreter.parsing import get_engine
from prolog.interpreter.continuation import Continuation, Engine, \
        DoneSuccessContinuation, DoneFailureContinuation
from prolog.interpreter import error, term
import prolog.interpreter.term
prolog.interpreter.term.DEBUG = False

helptext = """
 ';':   redo
 'p':   print
 'h':   help
 
"""

class StopItNow(Exception):
    pass

class ContinueContinuation(Continuation):
    def __init__(self, engine, var_to_pos, write):
        Continuation.__init__(self, engine, DoneSuccessContinuation(engine))
        self.var_to_pos = var_to_pos
        self.write = write

    def activate(self, fcont, heap):
        self.write("yes\n")
        var_representation(self.var_to_pos, self.engine, self.write, heap)
        while 1:
            if isinstance(fcont, DoneFailureContinuation):
                self.write("\n")
                return DoneSuccessContinuation(self.engine), fcont, heap
            res = getch()
            if res in "\r\x04\n":
                self.write("\n")
                raise StopItNow()
            if res in ";nr":
                raise error.UnificationFailed
            elif res in "h?":
                self.write(helptext)
            elif res in "p":
                var_representation(self.var_to_pos, self.engine, self.write, heap)
            else:
                self.write('unknown action. press "h" for help\n')

class GetAllScoresContinuation(Continuation):
    nextcont = None
    def __init__(self, engine, scores, similarity, depths, rules, unifications, min_width):
        Continuation.__init__(self, engine, DoneSuccessContinuation(engine))
        self.scores = scores
        self.similarity = similarity
        self.depths = depths
        self.rules = rules
        self.unifications = unifications
        self.min_width = min_width

    def activate(self, fcont, heap):
        width = 0
        for r in heap.rules:
            if ':-' in r:
                w = r.count("),") + 1

                if w > width:
                    width = w
        if width >= self.min_width:
            self.scores.append(heap.score())
            self.similarity.threshold = max(heap.score(), self.similarity.threshold)
            self.depths.append(heap.depth)
            self.rules.append(heap.rules)
            self.unifications.append(heap.unifications)
        raise error.UnificationFailed

def var_representation(var_to_pos, engine, write, heap):
    from prolog.builtin import formatting
    f = formatting.TermFormatter(engine, quoted=True, max_depth=20)
    for var, real_var in var_to_pos.iteritems():
        if var.startswith("_"):
            continue
        value = real_var.dereference(heap)
        val = f.format(value)
        if isinstance(value, term.AttVar):
            write("%s\n" % val)
        else:
            write("%s = %s\n" % (var, val))
        
def getch():
    line = readline()
    return line[0]

def debug(msg):
    os.write(2, "debug: " + msg + '\n')

def printmessage(msg):
    os.write(1, msg)

def readline():
    result = []
    while 1:
        s = os.read(0, 1)
        result.append(s)
        if s == "\n":
            break
        if s == '':
            if len(result) > 1:
                break
            raise SystemExit
    return "".join(result)

def run(query, var_to_pos, engine):
    #from prolog.builtin import formatting
    #f = formatting.TermFormatter(engine, quoted=True, max_depth=20)
    try:
        if query is None:
            return
        engine.run_query_in_current(
                query,
                ContinueContinuation(engine, var_to_pos, printmessage))
    except error.UnificationFailed:
        printmessage("Nein\n")
    except error.UncaughtError, e:
        printmessage("ERROR:\n%s\n" % e.format_traceback(engine))
    except error.CatchableError, e:
        printmessage("ERROR: %s\n" % e.get_errstr(engine))
    except error.PrologParseError, exc:
        printmessage(exc.message + "\n")
    # except error.UncatchableError, e:
    #     printmessage("INTERNAL ERROR: %s\n" % (e.message, ))
    except StopItNow:
        printmessage("yes\n")
def repl(engine):
    printmessage("welcome!\n")
    while 1:
        module = engine.modulewrapper.current_module.name
        if module == "user":
            module = ""
        else:
            module += ":  "
        printmessage(module + ">?- ")
        line = readline()
        if line == "halt.\n":
            break
        try:
            goals, var_to_pos = engine.parse(line, file_name="<stdin>")
        except error.PrologParseError, exc:
            printmessage(exc.message + "\n")
            continue
        for goal in goals:
            run(goal, var_to_pos, engine)

def execute(e, filename):
    run(term.Callable.build("consult", [term.Callable.build(filename)]), {}, e)

if __name__ == '__main__':
    from sys import argv
    e = Engine(load_system=True)
    if len(argv) == 2:
        execute(e, argv[1])
    repl(e)
