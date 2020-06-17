# sPyrolog
A Prolog interpreter with support for weak unification (two symbols can unify if they are sufficiently similar)

This project is a fork of sPyrolog is a fork of the [Prolog interpreter Pyrolog](https://bitbucket.org/cfbolz/pyrolog/).

## Build
sPyrolog is written in RPython and can be compiled with

```pypy-6.0.0-linux_x86_64-portable/bin/rpython targetprologstandalone.py -oJIT```

## Usage
While sPyrolog should support arbitrary Prolog programs, it has been written for use in [NLProlog](https://github.com/leonweber/nlprolog) and was only tested in this context.

This is highly experimental research code which is not suitable for production usage. We do not provide warranty of any kind. Use at your own risk.


### sPyrolog Programs
A sPyrolog program consists of two files:

The `facts` file contains the facts and rules of the Prolog program in the following syntax: `[statement] = [score]`.
A very simple example program would be:

```
country(X,Z) :- is_in(X,Y), country(Y,Z). = 1.0
country(berlin, germany). = 1.0
is_located_in(berlin, germany). = 0.8
is_located_in(X,Z) :- is_in(X,Y), country(Y,Z). = 0.8
```

The `similarity` file contains similarities between the symbols of the `facts` file:

```
country ~ is_located_in = 0.8
```

Note, that this implemenation requires that the weak unification for predicate symbols with rule heads and facts has to be done in a preprocessing step.
For instance, the similarity `country ~ is_located_in = 0.8` makes unification of `country` and `is_located_in` posssible.
For sPyrolog to respect this, the fact `is_located_in(berlin, germany). = 0.8` derived from `country(berlin, germany). = 1.0` has to be included in the `facts` file.
If two possible unifications lead to the same new fact/rule, only the one with the maximum score may be included.

### CLI
After generating the `facts` and the `similarity` file, sPyrolog can be called with the following parameters:
```
spyrolog FACTS_FILE SIMILARITY_FILE GOAL1|GOAL2|...|GOALN MAX_DEPTH LAMBDA_CUT ENTITY_TNORM|PREDICATE_TNORM MIN_RULE_WIDTH
```
* `FACTS_FILE`: path to the `facts` file as described above
* `SIMILARITY_FILE`: path to the `similarity` file as described above
* `GOAL1|GOAL2|...|GOALN`: `|`-separated goals for which to search for proofs, e.g. `is_located_in(berlin, germany).|country(berlin,germany).` will attempt to prove the two given statements
* `MAX_DEPTH`: The maximum proof depth
* `LAMBDA_CUT`: The threshold for the proof score. If the proof score falls below this threshold for a given subgoal, the proof for the subgoal fails.
* `ENTITY_TNORM`: The aggregation function for entity similarities. Has to be one of `{prod,min,luk}`.
* `PREDICATE_TNORM`: The aggregation function for predicate similarities. Has to be one of `{prod,min,luk}`.
* `MIN_RULE_WIDTH`: If this is greater than 0, then all proofs that do not use at least one rule with at least `MIN_RULE_WIDTH` antecedents are discarded. This is highly specific to the use case of NLProlog and probably should be set to 0.

As output, sPyrolog prints one line per proof goal. The i'th line describes the proof with the maximum score found for the i'th goal in the following syntax:
```
SCORE DEPTH UNIFICATION1|UNIFICATION2|...|UNIFICATIONN PROOF_TREE
```
* `SCORE`: The score of the proof
* `DEPTH`: The depth of the proof
* `UNIFICATION1|UNIFICATION2|...|UNIFICATIONN`: All unifications employed in the proof, e.g. `is_located_in<>country|contry<>country`
* `PROOF_TREE`: A linearization of the proof tree which can be delinearized with [this function](https://github.com/leonweber/nlprolog/blob/6b836ae2a03496fd55e963dd35877e55eac672a0/visualize_proof_tree.py#L56)
