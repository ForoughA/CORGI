"""
A simple standalone target for the prolog interpreter.
""" 
import py
import sys
from prolog.interpreter.translatedmain import GetAllScoresContinuation

# __________  Entry point  __________

from prolog.interpreter.continuation import Engine, jitdriver
from prolog.interpreter import term
from prolog.interpreter import arithmetic # for side effects
from prolog import builtin # for side effects
from prolog.interpreter.error import UnificationFailed
from prolog.interpreter.similarity import get_similarity_from_file

from rpython.rlib import jit
#from rpython import jit

term.DEBUG = False

def get_max_info(scores, depths, rules, unifications):
    max_score = 0.0
    max_depth = 0
    max_rule = []
    max_unification = []
    for i in range(len(scores)):
        score = scores[i]
        depth = depths[i]
        rule = rules[i]
        unification = unifications[i]

        if score > max_score:
            max_score = score
            max_depth = depth
            max_rule = rule
            max_unification = unification

    return max_score, max_depth, max_rule, max_unification


def entry_point(argv):
    if len(argv) != 8:
        print("Usage: spyrolog PROGRAM SIMILARITIES QUERY1|QUERY2|...|QUERYN MAX_DEPTH LAMBDA_CUT1|LAMBDA_CUT2|...|LAMBDA_CUTN E-TNORM|P-TNORM MIN_WIDTH")
        return 1

    program_fname = argv[1]
    sim_fname = argv[2]
    queries = argv[3]
    max_depth = int(argv[4])
    lambda_cuts = [float(i) for i in argv[5].split('|')]
    entity_tnorm, predicate_tnorm = argv[6].split('|')
    min_width = int(argv[7])

    sim = get_similarity_from_file(sim_fname, 0, entity_tnorm, predicate_tnorm)
    e = Engine(load_system=False, similarity=sim, max_depth=max_depth)
    e.modulewrapper.current_module = e.modulewrapper.user_module
    with open(program_fname) as f:
        sim.parse_rulescores(f.read(), e)

    query_idx = 0
    for query in queries.split('|'):
        cut = lambda_cuts[query_idx]
        scores = []
        depths = []
        rules = []
        unifications = []
        sim.lambda_cut = cut
        sim.reset_threshold()
        sim.query_idx = query_idx
        collector = GetAllScoresContinuation(e, scores, sim, depths, rules, unifications, min_width)
        e.max_depth = max_depth

        goals, var_to_pos = e.parse(query)
        goal = goals[0]

        try:
            e.run_query_in_current(goal, collector)
        except UnificationFailed:
            info = get_max_info(scores, depths, rules, unifications)
            if info[0] > 0:
                print(info[0], info[1], '|'.join(info[3]), query + ''.join(info[2]))
            else:
                print(0.0, 0)

        query_idx += 1
    return 0

# _____ Define and setup target ___


def target(driver, args):
    driver.exe_name = 'spyrolog'
    return entry_point, None

def portal(driver):
    from prolog.interpreter.portal import get_portal
    return get_portal(driver)

def jitpolicy(self):
    from rpython.jit.codewriter.policy import JitPolicy
    return JitPolicy()

if __name__ == '__main__':
    entry_point(sys.argv)
