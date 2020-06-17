import re
import copy
import spacy
import random
import json
import pickle
import numpy as np
import argparse
import logging

import py  
import sys 

parser = argparse.ArgumentParser(description="CORGI: a common-sense reasoning by instruction engine",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--user', type=str, default='dummy',
                    help='indicate your name to append to the path to save your log file. If this is not specified, the interaction will not be properly saved')

parser.add_argument('--engine', type=str, default='/testnet',
                    help='path to the prolog engine, switch to /spyrolog if you want to test the model without the rule embeddings')

parser.add_argument('--resume', type=int, default=0,
                    help='resume study starting at statement #X')

parser.add_argument('--verbose', type=bool, default=False,
                    help='weather to print out things')

args = parser.parse_args()
sys.path[0] += args.engine
prologPath = args.engine
VERBOSE = args.verbose

from prolog.interpreter.translatedmain import GetAllScoresContinuation
from prolog.interpreter.continuation import Engine, jitdriver
from prolog.interpreter import term
from prolog.interpreter import arithmetic 
from prolog.interpreter.error import CatchableError
from prolog import builtin 
from prolog.interpreter.error import UnificationFailed
from prolog.interpreter.similarity import get_similarity_from_file, Similarity
from prolog.interpreter.test.tool import collect_all, assert_true, assert_false
from prolog.interpreter.parsing import parse_query_term
from prolog.interpreter.test.test_builtin import test_assert_at_right_end, test_assert_retract
from prolog.interpreter.signature import Signature

from rpython.rlib import jit #download rpython

from StringIO import StringIO


# GLOBALS
WORLDS = {}
WEATHERS = ['bad', 'good', 'sun', 'rain', 'storm', 'wind', 'snow', 'fog', 'cloud', 'thunderstorm']
for w in WEATHERS:
    assert w not in WORLDS
    WORLDS[w] = 'weather'
OTHER_WEATHERS = ['weather', 'degree', 'temperature']
for w in OTHER_WEATHERS:
    assert w not in WORLDS
    WORLDS[w] = 'weather'
EMAILS = ['recipient', 'subject', 'body', 'sender', 'email', 'receive']
for e in EMAILS:
    assert e not in WORLDS
    WORLDS[e] = 'email'
CALENDAR = ['meeting', 'appointment', 'invite', 'work', 'talk', 'event', 'bill', 'visit']
for c in CALENDAR:
    assert c not in WORLDS
    WORLDS[c] = 'calendar'
ALARMS = ['set', 'sat', 'alarm', 'wake']
for a in ALARMS:
    assert a not in WORLDS
    WORLDS[a] = 'alarm'
MAP = ['navigate', 'navigation', 'route', 'map']
MAP_Location = ['gas', 'restaurant', 'grocery', 'home']
for m in MAP:
    assert m not in WORLDS
    WORLDS[m] = 'navigation'
for m in MAP_Location:
    assert m not in WORLDS
    WORLDS[m] = 'navigation'
REMIND = ['remind', 'notify', 'wake', 'tell', 'say', 'ask', 'know', 'make sure']

def parseStatment(statement, nlpToolKit):


    analyzedStatement = nlpToolKit(unicode(statement, "utf-8"))
    namedEntities = [(X.text, X.label_) for X in analyzedStatement.ents]
    corefClusters = []

    try:
        S  = re.compile(r"if (.+?) then", re.IGNORECASE)
        A  = re.compile(r".* then (.+?) because", re.IGNORECASE)
        G  = re. compile(r".* because (.*)", re.IGNORECASE)

        state  = re.match(S, statement).group(1)
        action = re.match(A, statement).group(1)
        goal   = re.match(G, statement).group(1)
        print 'state:', state
        logging.debug('state:{0}'.format(state))
        print 'action:', action
        logging.debug('action:{0}'.format(action))
        print 'goal:', goal
        logging.debug('goal:{0}'.format(goal))
        print('\n')

    except:
        if statement != "exit." or "exit":
            logging.error('make sure your statement follows the format if ... then ... because ...')
            print "\nERROR: make sure your statement follows the format if ... then ... because ...\n"
        return

    supportedAction = ''
    for remindIndicator in REMIND:
        A2 = re.compile(r"(.*)\b{0}\b(.*)\b".format(remindIndicator))
        act = re.match(A2, action)
        if not act:
            A2 = re.compile(r"(.*)\b{0}(.*)\b".format(remindIndicator))
            act = re.match(A2, action)
        if act:
            supportedAction = act.group(0)
            break
    if supportedAction == '':
    	logging.error('I currently only support reminding things. Please remind me to do something :)')
        print "\nI currently only support reminding things. Please remind me to do something :)\n"
        return

    action = supportedAction

    try:
        reverseGoals = ["don't","dont","do not", "cannot", "can't", "will not", "won't"]
        for rg in reverseGoals:
            if re.search(rg, goal) is not None:
            	logging.error('I currently don"t support negative goal statements. Please state what your goal is')
                print "\nI currently don't support negative goal statements. Please state what your goal is\n"
                return
    except:
        pass

    return [(state, action, goal), (analyzedStatement, corefClusters, namedEntities)]

def groundState(state, nlpToolKit, namedEntities, prolog):


    analyzedState = nlpToolKit(unicode(state,"utf-8"))
    namedEntitiesState = [(X.text, X.label_) for X in analyzedState.ents]

    namedEntitiesState = []
    for ne in namedEntities:
        if ne[0] in state:
            namedEntitiesState.append(ne)
    
    worlds = set()
    for indicator, WORLD in WORLDS.iteritems():
        pattern = re.compile(r"(\b{0}.*\b)".format(indicator))
        match = re.search(pattern, state)
        if re.search(pattern, state) is not None:
            worlds.add(WORLD) # what if no world is found?

    worlds = list(worlds)        


    [groundedState1, predicate1, arguments1] = getDependencyTuple(analyzedState)
    [groundedState2, predicate2, arguments2] = getDependencyTupleTraverse(analyzedState, namedEntitiesState, prolog)


    groundedStates = []
    for world in worlds:
        if world == 'weather':
            groundWeather = 'Weather'
            for weather in WEATHERS:
                pattern = re.compile(weather)
                if pattern.search(state):
                    groundWeather = weather 
                    break
            groundWeatherDate = 'Date'
            groundTemperature = 'Temp'
            if namedEntitiesState:
                ct1 = 0
                ct2 = 0
                for nePair in namedEntitiesState:
                    ne, tp = nePair
                    if tp == 'DATE' or tp == 'TIME':
                        groundWeatherDate = ne.lower()
                        ct1 += 1
                    elif tp == 'QUANTITY':
                        temp = re.compile(r'(\b[0-9][0-9]?[0-9]?\b)')
                        groundTemperature = re.search(temp,ne).group(0) 
                        ct2 += 1
            
            groundedState = '{0}({1}, {2}, {3})'.format(world, groundWeather, groundWeatherDate, groundTemperature) 
        elif world == 'email':
            for email in EMAILS:
                pattern = re.compile(email)
                if pattern.search(state):
                    groundEmail = email
                    break

            groundedState = '{0}({1})'.format(world, groundEmail) # TODO: this is incorrect we would like to extract the field not the field name
            

        elif world == 'alarm':
            groundAlarm = 'Time'
            groundPerson = 'Person'
            if namedEntitiesState:
                for nePair in namedEntitiesState:
                    ne, tp = nePair
                    if tp == 'DATE' or tp == 'TIME':
                        groundAlarm = ne.lower()
                    elif tp == 'PERSON':
                        groundPerson = ne.lower()
             
            groundedState = '{0}({1}, {2})'.format(world, groundPerson, groundAlarm)
        
        elif world == 'calendar': 
            for calendarEvent in CALENDAR:
                pattern = re.compile(calendarEvent)
                if pattern.search(state):
                    eventType = calendarEvent
                    break

            eventTime     = 'Time'
            eventPerson   = 'Person'
            eventLocation = 'Location'
            for nePair in namedEntitiesState:
                    ne, tp = nePair
                    if tp == 'DATE' or tp == 'TIME':
                        eventTime = ne.lower()
                    elif tp == 'PERSON':
                        eventPerson = ne.lower()
                    elif tp == 'GPE' or tp == 'ORG':
                        eventLocation = ne.lower()

            groundedState = '{0}({1}, {2}, {3}, {4})'.format(world, eventPerson, eventType, eventTime, eventLocation)


        elif world == 'navigation':
            destination = 'Destination'
            origin      = 'Origin'
            duration    = 'Time'
            for dest in MAP_Location:
                pattern = re.compile(dest)
                if pattern.search(state):
                    destination = dest
                    break
            for nePair in namedEntitiesState:
                ne, tp = nePair
                if tp == 'GPE' or tp == 'ORG':
                    origin = ne
                elif tp == 'TIME' or tp == 'DATE':
                    duration = ne
            groundedState = '{0}({1},{2},{3})'.format(world, origin, destination, duration)


        groundedStates.append(groundedState)

    if VERBOSE:
        print 'groundedStates:', groundedStates
    logging.debug('groundedState:{0}'.format(groundedStates))
    if len(worlds) == 0:
    	return [groundedState2]
    return groundedStates

def groundAction(action, nlpToolKit, namedEntities, prolog, predicateDict, typeDict):

    namedEntitiesAction = []
    for ne in namedEntities:
        if ne[0] in action:
            namedEntitiesAction.append(ne)

    for remindIndicator in REMIND:
        if remindIndicator == 'wake':
            act = action

        else:
            pattern = re.compile(r"(.*)\b{0}(.*)({1})\b(.*)\b".format(remindIndicator, 'to|that|""'))
            matchedAction = re.match(pattern, action)

            if matchedAction:
                if matchedAction.group(1):
                    pass
                act = matchedAction.group(4) 
                break

    act = act.lstrip(' ').rstrip(' ')

    analyzedAction = nlpToolKit(unicode(act, "utf-8"))

    [groundedAction, predicate, arguments] = getDependencyTuple(analyzedAction)
    [groundedActionNew, predicateNew, argumentsNew] = getDependencyTupleTraverse(analyzedAction, namedEntitiesAction, prolog)

    if VERBOSE:
        print 'groundedActionNew:', groundedActionNew

    matchedAction = matchGoalAgainstKB(groundedActionNew, predicateDict, prolog, typeDict)

    if matchedAction:
    	groundedActionNew = matchedAction
    	predicateNew = matchedAction.split('(')[0]
    	argumentsNew = groundedActionNew[groundedActionNew.find('(')+1 : groundedActionNew.find(')')].split(',')


    logging.debug('groundedAction:{0}'.format(groundedActionNew))
    return [groundedActionNew, predicateNew, argumentsNew]

def _groundGoal(goal, nlpToolKit, prolog):

    pattern = re.compile(r"\b(.*)\b({0})\b(.*)".format('want to |need to |I want |want to be able to|be able to|""'))
    matchedGoal = re.match(pattern, goal)
    if matchedGoal:

        matchedGoal = matchedGoal.group(1) + matchedGoal.group(3)

    else:
        matchedGoal = goal

    pattern = re.compile(r"\b(.*)\b({0})\b(.*)".format('make sure|ensure|""'))
    matchedGoal2 = re.match(pattern, matchedGoal)
    if matchedGoal2:
        matchedGoal = matchedGoal2.group(1) + matchedGoal2.group(3)


    matchedGoal = matchedGoal.lstrip(' ').rstrip(' ')
    analyzedGoal = nlpToolKit(unicode(matchedGoal, "utf-8"))
    namedEntitiesGoal = [(X.text, X.label_) for X in analyzedGoal.ents]
    goalCorefClusters = []

    rootToken = list(analyzedGoal.sents)

    [groundedGoal, predicate, arguments] = getDependencyTuple(analyzedGoal)
    [groundedGoalNew, predicateNew, argumentsNew] = getDependencyTupleTraverse(analyzedGoal, namedEntitiesGoal, prolog)

    logging.debug('groundedGoal:{0}'.format(groundedGoalNew))
    logging.debug('matchedGoal:{0}'.format(matchedGoal))
    return [matchedGoal, groundedGoalNew, predicateNew, argumentsNew]

def createRule(goal, groundedGoal, body, groundedBody, nlpToolKit, typeDict, prolog):
    analyzedGoal = nlpToolKit(goal)
    analyzedBody = nlpToolKit(body)

    sameStems = set()
    lemmaDict = {}
    for token in analyzedGoal:
        if token.lemma_ not in lemmaDict:
            lemmaDict[token.lemma_] = 1    
        else:
            logging.debug('several words of the same root in a single statement')
            print 'several words of the same root in a single statement'
    for token in analyzedBody:
        if token.lemma_ in lemmaDict:
            sameStems.add(token.lemma_)

    refinedGoalStack, refinedBodyStack = refineDependencyTuples([groundedGoal,groundedBody], sameStems, typeDict, prolog)

    return refinedGoalStack, refinedBodyStack

def findType(all_rules, goal):
    goalPredicate = goal.split('(')[0]
    headArgs = []
    for rule in all_rules:
        if ':-' in rule:
            head = rule.split(":-")[0].rstrip(' ')
            body = rule.split(":-")[1].rstrip('.').lstrip(' ')
            headPredicate = head.split('(')[0]
            if goalPredicate == headPredicate:
                headArgs = head.split('(')[1].split(')')[0].split(',')
                break
    return headArgs

def reorderArgsByType(arguments, referenceArgs, typeDict):
    correctArgs = ['' for _ in arguments]
    usedArgs = [0 for _ in arguments]
    numUsed = 0
    for ind, refArg in enumerate(referenceArgs):
        if not refArg.lstrip(' ')[0].isupper():
                correctArgs[ind] = arguments[ind].lstrip(' ').rstrip(' ')
                usedArgs[ind] = 1
                numUsed += 1
        for ind2, arg in enumerate(arguments):
            if usedArgs[ind2]!=1 and arg.lower() in typeDict:
                argType = typeDict[arg]
                if len(argType) == 1:
                    argType = argType[0].split('_')[0]
                else:
                    argType = argType[0].split('_')[0]

                if argType.lower() in refArg.lower():
                    correctArgs[ind] = arg.lstrip(' ').rstrip(' ')
                    usedArgs[ind2] = 1
                    numUsed += 1
                    break

    if numUsed < len(arguments):
        ctr0 = 0
        ctr1 = 0
        while ctr0 < len(correctArgs) and ctr1 < len(usedArgs):
            while correctArgs[ctr0] != '':
                ctr0 += 1
                if ctr0 == len(correctArgs):
                    break
            while usedArgs[ctr1] != 0:
                ctr1 += 1
                if ctr1 == len(usedArgs):
                    break
            if ctr0 < len(correctArgs) and ctr1 < len(usedArgs):
                correctArgs[ctr0] = arguments[ctr1].lstrip(' ').rstrip(' ')
                ctr0 += 1
                ctr1 += 1

    for cArg in correctArgs:
        if cArg == '':
            logging.error('YOUR CODE IS NOT WORKING AS EXPECTED FIX YOUR REORDERING')
            print 'ERROR: YOUR CODE IS NOT WORKING AS EXPECTED FIX YOUR REORDERING'
    
    return correctArgs

def refineDependencyTuples(goalStack, bodyStack, typeDict, prolog):

    assert len(goalStack) == len(bodyStack), 'there should be the same number of goals and bodies'

    refinedGoalStack = []
    refinedBodyStack = []
    addedToGoal = []
    addedToBody = []

    all_rules = prolog.modulewrapper.current_module.get_all_rules()
    existingGoal = bodyStack[-1]
    typedArgs = findType(all_rules, existingGoal)
    numReferenceArgs = len(existingGoal[existingGoal.find('(')+1 : existingGoal.find(')')].split(','))
    for tup in zip(goalStack, bodyStack):
        currGoal, currBody = tup

        goalPredicate = currGoal.split('(')[0]
        bodyPredicate = currBody.split('(')[0]

        goalArguments = currGoal[currGoal.find('(')+1 : currGoal.find(')')].split(',')
        bodyArguments = currBody[currBody.find('(')+1 : currBody.find(')')].split(',')

        addedGoalArgs = [0 for _ in range(numReferenceArgs)]
        addedBodyArgs = [0 for _ in range(numReferenceArgs)]
        goalDiff = len(goalArguments) - numReferenceArgs
        if goalDiff > 0:
            goalVarArgs = [ind for ind, arg in enumerate(goalArguments) if arg[0].isupper()]
            for ctr in range(goalDiff):
                if ctr < len(goalVarArgs):
                    index = goalVarArgs[-1-ctr]
                    del(goalArguments[index])
        elif goalDiff < 0:
            startInd = len(goalArguments)-1
            for ctr in range(-goalDiff):
                startInd += 1
                if startInd < len(typedArgs):
                    goalArguments.append('{0}'.format(typedArgs[startInd].lstrip(' ').rstrip(' ')))
                    addedGoalArgs[startInd] = 1
        reorderedGoalArguments = reorderArgsByType(goalArguments, typedArgs, typeDict)
        
        refinedGoalStack.append('{0}({1})'.format(goalPredicate, ','.join(reorderedGoalArguments)))
        addedToGoal.append(addedGoalArgs)

        bodyDiff = len(bodyArguments) - numReferenceArgs
        if bodyDiff > 0:
            bodyVarArgs = [ind for ind, arg in enumerate(bodyArguments) if arg[0].isupper()]
            for ctr in range(bodyDiff):
                if ctr < len(bodyVarArgs):
                    index = bodyVarArgs[-1-ctr]
                    del(bodyArguments[index])
        elif bodyDiff < 0 :
            startBodyInd = len(bodyArguments)-1
            for ctr in range(-bodyDiff):
                startBodyInd += 1
                if startBodyInd < len(typedArgs):
                    bodyArguments.append('{0}'.format(typedArgs[startBodyInd]))
                    addedBodyArgs[startBodyInd] = 1

        reorderedBodyArguments= reorderArgsByType(bodyArguments, typedArgs, typeDict)
        refinedBodyStack.append('{0}({1})'.format(bodyPredicate, ','.join(reorderedBodyArguments)))
        addedToBody.append(addedBodyArgs)

    assert len(goalStack) == len(refinedGoalStack)
    assert len(bodyStack) == len(refinedBodyStack)
    assert len(addedToGoal) == len(refinedGoalStack)
    assert len(addedToBody) == len(refinedBodyStack)

    goalStack = refinedGoalStack
    bodyStack = refinedBodyStack
    refinedGoalStack = []
    refinedBodyStack = []
    for tup in zip(goalStack, bodyStack, addedToGoal, addedToBody):
        currGoal, currBody, currAddedToGoal, currAddedToBody = tup

        goalPredicate = currGoal.split('(')[0]
        bodyPredicate = currBody.split('(')[0]

        goalArguments = currGoal[currGoal.find('(')+1 : currGoal.find(')')].split(',')
        bodyArguments = currBody[currBody.find('(')+1 : currBody.find(')')].split(',')
        variableCtr = 0
        for goalInd, goalArg in enumerate(goalArguments):
            if goalArg[0].islower():
                try:
                    bodyInd = bodyArguments.index(goalArg)
                    argType = ''
                    if goalArg in typeDict:
                    	argType = typeDict[goalArg][0].split('_')[0]
                    if argType:
                    	argType = argType.capitalize()
                    	varName = '{0}{1}'.format(argType, variableCtr)
                    else:
                    	varName = 'Y{0}'.format(variableCtr)
                    variableCtr += 1
                    goalArguments[goalInd] = varName
                    bodyArguments[bodyInd] = varName
                except ValueError:
                    continue
            elif goalArg[0].isupper():
                try:
                    bodyInd = bodyArguments.index(goalArg)
                    pass

                except ValueError:
                    continue

        for indG, addG in enumerate(currAddedToGoal):
            addB = currAddedToBody[indG]
            if addG == 1 and addB == 0:
                assert goalArguments[indG][0].isupper()
                Btype = ''
                if addB in typeDict:
                    Btype = typeDict[addB][0].split('_')[0]
                if Btype and Btype in goalArguments[indG].lower():
                    goalArguments[indG] = bodyArguments[indG]

        refinedGoal = '{0}({1})'.format(goalPredicate, ','.join(goalArguments))
        refinedBody = '{0}({1})'.format(bodyPredicate, ','.join(bodyArguments))

        refinedGoalStack.append(refinedGoal)
        refinedBodyStack.append(refinedBody)

    assert len(refinedGoalStack) == len(goalStack)
    assert len(refinedBodyStack) == len(bodyStack)
    
    return [refinedGoalStack, refinedBodyStack]

def groundGoal(goal, predicateDict, nlpToolKit, namedEntities, prolog, typeDict, force_input=False):

    [cleanedUpGoal, groundedGoal, goalPredicate, goalArguments] = _groundGoal(goal, nlpToolKit, prolog)
    goalPredicate = groundedGoal.split('(')[0]
    matchedGoal = matchGoalAgainstKB(groundedGoal, predicateDict, prolog, typeDict)
    mainGoal = matchedGoal

    goalStack = []
    bodyStack = []
    goalPredicateStack = []
    bodyPredicateStack = []
    goalArgumentsStack = []
    bodyArgumentsStack = []
    numTries = 0

    while (force_input or not matchedGoal) and numTries < 3:
    	if force_input == True:
    		force_input = False

        goalStack.append(groundedGoal)
        goalPredicateStack.append(goalPredicate)
        goalArgumentsStack.append(goalArguments)
        logging.info('How do I know if "{0}"?'.format(cleanedUpGoal))
        newGoal = raw_input('\nHow do I know if "{0}"? \n'.format(cleanedUpGoal))
        logging.info('entered statement: {0}'.format(newGoal))
        [cleanedNewGoal, groundedBody, bodyPredicate, bodyArguments] = _groundGoal(newGoal, nlpToolKit, prolog)
        bodyStack.append(groundedBody)
        bodyPredicateStack.append(bodyPredicate)
        bodyArgumentsStack.append(bodyArguments)

        groundedGoal = groundedBody
        cleanedUpGoal = cleanedNewGoal

        matchedGoal = matchGoalAgainstKB(groundedGoal, predicateDict, prolog, typeDict)

        print 'matchedGoal2:',matchedGoal

        numTries += 1

    if not matchedGoal:
        logging.error("I was not able to perform reasoning by asking the user :(")
        print "\n--------------"
        print "ERROR: I was not able to perform reasoning by asking the user :( "
        print "--------------\n"
        return [[],[],[]]
    if bodyStack:
        bodyStack[-1] = matchedGoal

    logging.debug('mainGoal: {0}'.format(mainGoal))
    logging.debug('goalStack: {0}'.format(goalStack))
    logging.debug('bodyStack: {0}'.format(bodyStack))

    return [mainGoal, (goalStack, goalPredicateStack, goalArgumentsStack), (bodyStack, bodyPredicateStack, bodyArgumentsStack)]

def matchGoalAgainstKB(groundedGoal, predicateDict, prolog, typeDict):

    all_rules = prolog.modulewrapper.current_module.get_all_rules()
    typedArgs = findType(all_rules, groundedGoal)
    goalPredicate = groundedGoal.split('(')[0]
    arguments = groundedGoal[groundedGoal.find('(')+1 : groundedGoal.find(')')].split(',')
    varArgs = [ind for ind, arg in enumerate(arguments) if arg[0].isupper()]

    if goalPredicate in predicateDict:
        numArgs = len(typedArgs)  
        diff = len(arguments) - int(numArgs)
        if diff == 0:
            newGroundedGoal = groundedGoal
        elif diff > 0 : 
            for iter in range(diff):
                if varArgs:
                    index = varArgs[-1-iter]
                    del(arguments[index])
                else:
                    pass
            reorderedArguments = reorderArgsByType(arguments, typedArgs, typeDict)
            newGroundedGoal = '{0}({1})'.format(predicateDict[goalPredicate][1], ','.join(arguments))
        elif diff < 0:
            startInd = len(arguments) - 1
            for iter in range(-diff):
                startInd += 1
                if startInd < len(typedArgs):
                    arguments.append('{0}'.format(typedArgs[startInd].lstrip(' ').rstrip(' ')))

            reorderedArguments = reorderArgsByType(arguments, typedArgs, typeDict)
            newGroundedGoal = '{0}({1})'.format(predicateDict[goalPredicate][1], ','.join(arguments))
        return newGroundedGoal

    return False

def getDependencyTupleTraverse(analyzedStatement, namedEntities, prolog):

    def removeRepeats(lst):
        seen = set()
        seen_add = seen.add
        noRepeats = [x for x in lst if not (x in seen or seen_add(x))]
        return noRepeats

    predicate = ''
    arguments = []
    statementObj = list(analyzedStatement.sents)[0]
    rootToken = statementObj.root
    if rootToken.pos_ == 'VERB':
        predicate = rootToken.lemma_.lower()        
    hasVerb = 0 
    iterateArgs = ['','','','','','']
    for token in analyzedStatement:
        if token.pos_ == 'VERB':
            hasVerb = 1
        if predicate == '':
            if token.pos_ == 'VERB' and token.dep_ != 'aux':
                predicate = token.lemma_.lower()
        if 'obj' in token.dep_: 
            if iterateArgs[1] == '':
                iterateArgs[1] = token.text.lower()
            else:
                if token.dep_ == 'dobj': 
                    iterateArgs[1] = token.text.lower()
        elif 'subj' in token.dep_:
            iterateArgs[0] = token.text.lower()
        elif 'comp' in token.dep_:
            iterateArgs[2] = token.text.lower()
        elif 'mod' in token.dep_:
            iterateArgs[3] = token.text.lower()
        elif 'conj' in token.dep_:
            iterateArgs[4] = token.text.lower()
        elif 'oprd' in token.dep_:
            iterateArgs[5] = token.text.lower()

    if predicate == '':
        for token in analyzedStatement:
            if hasVerb and token.pos_ == 'VERB':
                predicate = token.lemma_.lower()
                break
            elif (not hasVerb) and token.dep_ == 'ROOT':
                predicate = token.text.lower()
                break

    for child in rootToken.children:
        if child.pos_ != 'PART' and child.pos_ not in ['PUNC','SPACE']:
            ctext = child.text.lower()
            arguments.append(ctext)
        else:
            pass

    spacyArguments = []
    for chunk in analyzedStatement.noun_chunks:
        np = chunk.text.lower().replace(' ', '_')
        if chunk.root.head.text.lower() != predicate and \
           chunk.root.head.pos_ != 'VERB'            and \
           chunk.root.head.text.lower() not in np:
            np = chunk.root.head.text.lower().replace(' ', '_')+'_'+ np

        spacyArguments.append(np)

    for arg in iterateArgs:
        if arg != predicate and not any([arg in sargs for sargs in spacyArguments]):
            spacyArguments.append(arg)
    spacyArguments = removeRepeats(spacyArguments)
    for ne in namedEntities:
        used = 0
        for ind, arg in enumerate(spacyArguments):
            if arg in ne[0] or ne[0] in arg:
                used = 1
                break
        if used == 0 and ne[0] not in spacyArguments:
            spacyArguments.append(ne[0].replace(' ','_'))
    spacyArguments = removeRepeats(spacyArguments)

    if VERBOSE:
        print('spacyArguments: {0}'.format(spacyArguments))
    logging.debug('spacyArguments: {0}'.format(spacyArguments))
    sig = Signature.getsignature(str(predicate), len(spacyArguments))
    isBuiltin = prolog.get_builtin(sig) 
    if isBuiltin or str(predicate)=='close' or str(predicate)=='open':
    	predicate = predicate + '_new'
    statementTupleSpacy = '{0}({1})'.format(predicate, ','.join(spacyArguments))
    if VERBOSE:
        print 'statementTupleSpacy:', statementTupleSpacy

    statementTuple = '{0}({1})'.format(predicate, ','.join(arguments))
    if predicate == '':
    	logging.error('predicate name cannot be empty, debugging needed')
    assert predicate, 'predicate name cannot be empty, debugging needed'
    return [statementTupleSpacy, predicate, spacyArguments]

def getDependencyTuple(analyzedStatement):

    predicate = ''
    arguments = ['','','','','']
    depType = ['','','','','']
    hasVerb = 0
    for token in analyzedStatement:
        if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
            if predicate != '':
                logging.error('cannot have two verb ROOTs')
            assert predicate == '', 'cannot have two verb ROOTs'
            predicate = token.lemma_.lower()
        if token.pos_ == 'VERB':
            hasVerb = 1
        elif 'obj' in token.dep_: 
            if arguments[1] == '':
                arguments[1] = token.text.lower()
                depType[1] = token.dep_
            else:
                if token.dep_ == 'dobj' and depType[1] != 'dobj':
                    arguments[1] = token.text.lower()
                    depType[1] = token.dep_
        elif 'subj' in token.dep_:
            arguments[0] = token.text.lower()
            depType[0] = token.dep_
        elif 'comp' in token.dep_:
            arguments[2] = token.text.lower()
            depType[2] = token.dep_
        elif 'mod' in token.dep_:
            arguments[3] = token.text.lower()
            depType[3] = token.dep_
        elif 'conj' in token.dep_:
            arguments[4] = token.text.lower()
            depType[4] = token.dep_

    if predicate == '':
        if hasVerb:
            for token in analyzedStatement:
                if token.pos_ == 'VERB':
                    predicate = token.lemma_.lower()
                    break
        else:
            for token in analyzedStatement:
                if token.dep_ == 'ROOT':
                    predicate = token.text.lower()
                    break

    if arguments[0] == '':
        arguments[0] = 'Person'
    if arguments[1] == '':
        arguments[1] = 'What'
    if arguments[2] == '':
        arguments[2] = 'How'
    if arguments[3] == '':
        arguments[3] = 'State'
    if arguments [4] == '':
        del(arguments[4])

    if len(arguments) == 4:
        statementTuple = '{0}({1},{2},{3},{4})'.format(predicate, arguments[0], arguments[1], arguments[2], arguments[3])
    else:
        statementTuple = '{0}({1},{2},{3},{4},{5})'.format(predicate, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4])

    return [statementTuple, predicate, arguments]
    
def addInstructedRules(goalStack, bodyStack, prolog, predicateDict, sim):

    if len(goalStack) != len(bodyStack):
    	logging.error('there should be as many goals as there are bodys')
    assert len(goalStack) == len(bodyStack), 'there should be as many goals as there are bodys'
    while goalStack:
        currGoal = goalStack.pop()
        currBody = bodyStack.pop()
        goalPredicate = currGoal.split('(')[0]
        bodyPredicate = currBody.split('(')[0]
        goalArguments = currGoal[currGoal.find('(')+1 : currGoal.find(')')].split(',')
        bodyArguments = currBody[currBody.find('(')+1 : currBody.find(')')].split(',')
        if goalPredicate.lower() not in predicateDict:
            predicateDict[goalPredicate.lower()] = [len(goalArguments), goalPredicate]
            with open("functor_arity_instructed.txt", 'a') as textFile:
                textFile.write('{0}/{1}\n'.format(goalPredicate, len(goalArguments)))
        if bodyPredicate.lower() not in predicateDict:
            predicateDict[bodyPredicate.lower()] = [len(bodyArguments), bodyPredicate]
            with open("functor_arity_instructed.txt", 'a') as textFile:
                textFile.write('{0}/{1}\n'.format(bodyPredicate, len(bodyArguments)))

        newRule = currGoal + ':-' + currBody + '.'

        logging.info('newRule: {0}'.format(newRule))
        print "newRule", newRule
        newRuleWithScore = newRule + ' = 1.0'

        sim.parse_rulescores(newRuleWithScore, prolog)
        with open("facts4.txt", 'a') as ruleFile:
            ruleFile.write('{0} \n'.format(newRuleWithScore))

    return prolog, predicateDict

def isFactInKB(factPredicate, factArgs, factQuery, prolog, predicateDict):
    sig = Signature.getsignature(str(factPredicate), len(factArgs))
    isBuiltin = prolog.get_builtin(sig) 
    assert (not isBuiltin), 'we should have already checked this'

    factInKB = False
    
    query = '{0}({1}).'.format(factPredicate, ','.join(['X_{0}'.format(i) for i in range(len(factArgs))]))
    if prologPath == '/testnet' or prologPath == '/proofspyrolog':
        proof = collect_all(prolog, query, [], [], [], [])
    elif prologPath == '/spyrolog':
    	proof = collect_all(prolog, query, [], [], [])
    
    for vGrounding in proof:
        groundings = vGrounding[0]
        args = set()
        for variable, grounding in groundings.iteritems():
            if hasattr(grounding, 'name'):
                args.add(grounding.name())
            elif hasattr(grounding, 'num'):
                args.add(grounding.num)
            else:
                print 'dir:', dir(grounding)
                raise AttributeError

        if not set(factArgs).difference(args):
            factInKB = True
            break

    return factInKB

def addFactToKb(prolog, sim, query, factPredicate, factArgs, predicateDict):
    addingToPredicateDict = 0
    simQuery = query + ' = 1.0'
    sim.parse_rulescores(simQuery, prolog)
    if factPredicate not in predicateDict:
        predicateDict[factPredicate] = [len(factArgs), factPredicate]
        addingToPredicateDict = 1
    else:
        if int(predicateDict[factPredicate][0]) != len(factArgs):
            predicateDict[factPredicate+'_artificiallyAdded'] = [len(factArgs), factPredicate]
            addingToPredicateDict = 2
    return (prolog, sim, addingToPredicateDict)

def removeFactFromKb(prolog, query, factPredicate, addingToPredicateDict, predicateDict):
    query = 'retract({0}).'.format(query[:-1])
    terms, vars = prolog.parse(query)
    term, = terms
    prolog.run_query_in_current(term)

    if addingToPredicateDict == 1:
        del(predicateDict[factPredicate])
    elif addingToPredicateDict == 2:
        del(predicateDict[factPredicate+'_artificiallyAdded'])

    return prolog

def proveStatement(state, action, goal, prolog, predicateDict, nlpToolKit, sim, analyzedStatement, typeDict):

    def isPredInRule(predicate, rule):
        isIn = False
        for r in rule:
            if predicate in r:
                isIn = True
                break
        return isIn

    analyzedStatement, corefClusters, namedEntities = analyzedStatement
    stackFlag = False
    proof = []
    force_input = False
    groundedStates = groundState(state, nlpToolKit, namedEntities, prolog)
    statePredicates = []
    stateArguments = []
    for groundedState in groundedStates:
        statePredicates.append(groundedState.split('(')[0])
        stateArguments.append(groundedState[groundedState.find('(')+1 : groundedState.find(')')].split(','))
    [groundedAction, actionPredicate, actionArguments] = groundAction(action, nlpToolKit, namedEntities, prolog, predicateDict, typeDict)
    addingQuery = groundedAction + '.'
    actionInKB = isFactInKB(actionPredicate, actionArguments, addingQuery, prolog, predicateDict)
    if not actionInKB:
        [prolog, sim, addingToPredicateDict] = \
                addFactToKb(prolog, sim, addingQuery, actionPredicate, actionArguments, predicateDict)

    while not proof and not stackFlag:
        
        [groundedGoal, goalStack, bodyStack] = groundGoal(goal, predicateDict, nlpToolKit, namedEntities, prolog, typeDict, force_input)
        if not goalStack: break
        goalStack, goalPredicateStack, goalArgumentsStack = goalStack

        if len(goalStack) > 0:
            stackFlag = True

        for goalArgs in goalArgumentsStack:
            [gArg.replace('_', ' ') for gArg in goalArgs]
        bodyStack, bodyPredicateStack, bodyArgumentsStack = bodyStack

        if groundedGoal == False and goalStack == []:
            logging.error('ERROR: goal not grounded and goal stack empty. No goal to prove!!!')
            print "ERROR: goal not grounded and goal stack empty. No goal to prove!!!"
            return
        if (not groundedGoal) or force_input :
            goalStack, bodyStack = refineDependencyTuples(goalStack, bodyStack, typeDict, prolog)
            groundedGoal = goalStack[0]
            print "\n___________ADDING RULE__________\n"
            [prolog, predicateDict] = addInstructedRules(goalStack, bodyStack, prolog, predicateDict, sim)

        queryString = groundedGoal + '.'
        newQueryString, var_to_pos = prolog.parse(queryString)
        logging.info('queryString: {0}'.format(queryString))
        print 'queryString:', queryString

        query = newQueryString[0]

        logging.info('___________RUNNING QUERY__________,\n query: {0}'.format(query))
        print "___________RUNNING QUERY__________\n\n", "query", query

        scores = []
        depths = []
        rules = []
        unifications = []
        queries = []
        try:
            if prologPath == '/testnet' or prologPath == '/proofspyrolog':
                collect = collect_all(prolog, queryString, rules, scores, unifications, queries)
            if prologPath == '/spyrolog':
                collect = collect_all(prolog, queryString, rules, scores, unifications)
            logging.info('variable bindings: {0}'.format(collect))
            print "\n"
            print "variable bindings", collect
            print "\n"
            print("__________PROOF TRACE___________\n")
            logging.info('__________PROOF TRACE___________')
            proof = collect
            logging.info('all proofs: {0}'.format(proof))

        except Exception as e:
            logging.info('PROOF FAILED :( {0}'.format(e))
            print "proof failed"
            proof = None
        force_input = not stackFlag

    acceptedProof = []
    if proof:
        assert len(proof) == len(rules), 'the length of the proof and the trace should be the same'
        for p, r in zip(proof, rules):
            actionUsed = isPredInRule(actionPredicate, r)
            if actionUsed: 
                acceptedProof.append(p[0])
    print 'acceptedProof:', acceptedProof
    for p in acceptedProof:
       logging.info('acceptedProof: {0}', p)
       print 'p:', p

    if not actionInKB:
    	prolog = removeFactFromKb(prolog, addingQuery, actionPredicate, addingToPredicateDict, predicateDict)
    return acceptedProof


def logAndPrint(text):
    logging.info(text)
    print text

def generateUtterance(proof, state, action, goal):
    utterances = []
    if not proof:
        utterances.append('sorry, i do not know how to do what you asked me :( ')

    else:
        for p in proof:
            utterance = 'I will perform "{0}" in order to achieve "{1}" '.format(action, goal)
            if p:
                string = 'and make sure I'
                for variable, grounding in p.iteritems():
            	    string = string + ' set ' + variable + ' to ' + str(grounding) + ' and '
                string = string.rstrip('and ')
                utterance = utterance + string
            utterances.append(utterance)


    return utterances



def main():
    print "please stand by while the model loads..."
    cmdArg = parser.parse_args()
    user = cmdArg.user
    
    resume = cmdArg.resume
    logPath = 'logging/{0}.log'.format(user)


    logging.basicConfig(filename=logPath,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    logging.info('...starting session... user {0}'.format(user))

    with open('prolog_info-reasoning-17.pkl', 'rb') as f:
        rule_embeddings, var_embeddings, rule2index_dict, var2index_dict = \
        pickle.load(f)


    rule_embeddings = np.reshape(rule_embeddings, (-1, 256))
    if prologPath == '/testnet':
        sim = Similarity(0.6, var2index = var2index_dict,
                     embeddings = var_embeddings)
        prolog = Engine(load_system = False, similarity = sim, 
                    embeddings = rule_embeddings, rule2index = rule2index_dict)
    elif prologPath == '/spyrolog' or prologPath == '/proofspyrolog':
        sim = get_similarity_from_file('sim7.txt', 0.1, 'prod', 'prod')    
        prolog = Engine(load_system = False, similarity = sim)
        prolog.modulewrapper.current_module = prolog.modulewrapper.user_module


    with open('facts7.txt') as f:
        sim.parse_rulescores(f.read(), prolog) 

    with open("typeDict.json", "r") as jFile:
    	typeDict = json.load(jFile)

 
    nlp = spacy.load('en_coref_lg')
    for word in nlp.Defaults.stop_words:
        lex = nlp.vocab[word]
        lex.is_stop = True
    with open("functor_arity7.txt") as txtFile:
        listOfPredicates = txtFile.readlines()

    listOfPredicates = [(p.rstrip('\n').split('/')) for p in listOfPredicates]
    predicateDict = {p[0].lower():[p[1], p[0]] for p in listOfPredicates}
    with open('user_study_data.txt') as dataFile:
        statements = dataFile.readlines()
    
    for i_statement,statement in enumerate(statements):
        if statement[0] == '%':
            continue
        if i_statement<resume:
            continue
        statement = statement.rstrip('\n').rstrip('.')
        statement = statement + '.'
        if statement != 'exit.' or 'exit':
            logAndPrint('statement {1}: {0}\n'.format(statement,i_statement))

            ([state, action, goal], analyzedStatement) = parseStatment(statement, nlp)
            proof = proveStatement(state, action, goal, prolog, predicateDict, nlp, sim, analyzedStatement, typeDict)
            utterances = generateUtterance(proof, state, action, goal)
            print "\n--------------"
            if proof and len(proof) > 1:
                logging.info("There are multiple possibilities. please let me know which one you prefer:")
                print "There are multiple possibilities. please let me know which one you prefer:\n"
            for iiind, utterance in enumerate(utterances):
                print iiind, ':', utterance
                logging.info('returned proofs: {1} : {0}'.format(utterances, iiind))

            if len(utterances) > 1:
                selection = raw_input('select entry and hit enter:')
                logging.info('selected entry: {0}'.format(selection))

            print "--------------\n"

            logging.info('press any key to continue to the next statement\n')
            raw_input('press any key to continue to the next statement')
            print "--------------\n"

        else:
            break 
    print "\nStudy complete! Thank you\n"
    print "\nquitting...\n"
    logging.info('...ending session...')

if __name__=='__main__':
    main()
