
:- module(jobsquery, [
	answer/2,
	execute_query/2,
	print_name/2,
	title/2,
	company/2,
	recruiter/2,
	req_exp/2,
	des_exp/2,
	req_deg/2,
	des_deg/2,
	loc/2,
	salary_less_than/2,
	salary_less_than/3,
	salary_greater_than/2,
	salary_greater_than/3,
	req_exp/1,
	des_exp/1,
	req_deg/1,
	des_deg/1]).


job('89ljtl$ig6$267@news.gate.net',computer_science,'Software Engineer','n/a','n/a', 2,'n/a','n/a','n/a','2/03/00').


job(ID) :- job(ID,_,_,_,_,_,_,_,_,_).
title(ID,T) :- job(ID,_,T,_,_,_,_,_,_,_).
company(ID,C) :- job(ID,_,_,C,_,_,_,_,_,_).
recruiter(ID,R) :- job(ID,_,_,_,R,_,_,_,_,_).
req_exp(ID,E) :- job(ID,_,_,_,_,E,_,_,_,_).
des_exp(ID,E) :- job(ID,_,_,_,_,_,E,_,_,_).
req_deg(ID,D) :- job(ID,_,_,_,_,_,_,D,_,_).
des_deg(ID,D) :- job(ID,_,_,_,_,_,_,_,D,_).
post_date(ID,D) :- job(ID,_,_,_,_,_,_,_,_,D).


req_deg(ID) :- req_deg(ID, D), D \= 'n/a'.
req_exp(ID) :- req_exp(ID, E), E \= 'n/a'.
des_deg(ID) :- des_deg(ID, D), D \= 'n/a'.
des_exp(ID) :- des_exp(ID, E), E \= 'n/a'.


salary_greater_than(Job,Num) :- salary_greater_than(Job,Num,year).
salary_greater_than(Job,Num,Time) :-
	var(Num),
	salary(Job,S,M,Time),
	number(S),
	(S>=Num).
salary_less_than(Job,Num) :- salary_less_than(Job,Num,year).
salary_less_than(Job,Num,Time) :-
	salary(Job,S,M,Time),
	number(S),
	(S=<Num).
salary_range(Job,Lower,Upper,Time) :-
	salary_greater_than(Job,Lower,Time),
	salary_less_than(Job,Upper,Time).
salary_range(Job,Lower,Upper) :-
	salary_range(Job,Lower,Upper,year).

exp(ID,E) :- req_exp(ID,E);des_exp(ID,E).

exp_greater_than(J,N) :-
	exp(J,S),
	number(S),
	(S>=N).
exp_less_than(J,N) :-
	exp(J,S),
	number(S),
	(S=<N).
exp_range(J,Lower,Upper) :-
	exp_greater_than(J,Lower),
	exp_less_than(J,Upper).

req_exp(ID,'=<',Exp) :- 
	req_exp(ID,E),
	number(E),
	Exp =< E.
req_exp(ID,'=',Exp) :-
	req_exp(ID,Exp).
req_exp(ID,'>=',Exp) :-
	req_exp(ID,E),
	number(E),
	Exp >= E.

des_exp(ID,'=<',Exp) :-
	des_exp(ID, E),
	number(E),
	Exp =< E.
des_exp(ID,'=',Exp) :-
	des_exp(ID,Exp).
des_exp(ID,'>=',Exp) :-
	des_exp(ID,E),
	number(E),
	Exp >= E.


abbreviation(stateid(State), Ab) :- 
	state(_,State,Ab).
abbreviation(Ab) :- abbreviation(_,Ab).

print_name(stateid(X),X) :- !.
print_name(cityid(X,_), X) :- !.
print_name(placeid(X), X) :- !.
print_name(Goal, Y) :- (Goal=_/_;Goal=_*_;Goal=_+_;Goal=_-_),!, Y is Goal.
print_name(X,X).

%jobs meaning of loc, meaning a job is in a place:
loc(X, Place) :- city(X, Place).

first(G) :- (G -> true).

n_solutions(N,Goal) :-
	findall(Goal, Goal, GList0),
	length(Solutions, N),
	append(Solutions,_,GList0),
	member(Goal, Solutions).

nth_solution(N,Goal) :-
	findall(Goal, Goal, GList),
	nth(N,GList,Goal).

len(riverid(R), L) :-
	river(R,L,_).

size(stateid(X), S) :-
	area(stateid(X), S).
size(cityid(X,St), S) :-
	population(cityid(X,St), S).
size(riverid(X), S) :-
	len(riverid(X),S).
size(placeid(X), S) :-
	elevation(placeid(X),S).
size(X,X) :-
	number(X).
	
largest(Var, Goal) :-
	findall(Size-Goal, (Goal,size(Var,Size)), Pairs0),
	max_key(Pairs0, Goal).

max_key([Key-Value|Rest],Result) :-
	max_key(Rest, Key, Value, Result).

max_key([], _, Value, Value).
max_key([K-V|T], Key, Value, Result):-
	( K > Key ->
	     max_key(T, K, V, Result)
	; max_key(T, Key, Value, Result)
	).

smallest(Var, Goal) :-
	findall(Size-Goal, (Goal,size(Var,Size)), Pairs0),
	min_key(Pairs0, Goal).

min_key([Key-Value|Rest],Result) :-
	min_key(Rest, Key, Value, Result).

min_key([], _, Value, Value).
min_key([K-V|T], Key, Value, Result):-
	( K < Key ->
	     min_key(T, K, V, Result)
	; min_key(T, Key, Value, Result)
	).
at_least(N, S) :-
    number(N),
    number(S),
    N >= S.

at_most(N, S) :-
    number(N),
    number(S),
    N =< S.

/*CAT, following from geoquery
count(V,Goal,N) :-
	findall(V,Goal,Ts),
	sort(Ts, Unique),
	length(Unique, N).

at_least(Min,V,Goal) :-
	count(V,N,Goal),
	Goal,  % This is a hack to instantiate N, making this order independent.
	N >= Min.

at_most(Max,V,Goal) :-
	count(V,Goal,N),
	N =< Max.
*/

execute_query(Query, Unique):-
	tq(Query, answer(Var,Goal)),
	(Goal = freevar ->
	 Unique = []
	; findall(Name, (Goal, print_name(Var,Name)), Answers),
	  sort(Answers, Unique)
	).
%---------------------------------------------------------------------------
tq(G,G) :-
	var(G), !.
tq(largest(V,Goal), largest(Vars, DVars, DV, DGoal)) :-
	!,
	variables_in(Goal, Vars),
	copy_term((Vars,V,Goal),(DVars,DV,Goal1)),
	tq(Goal1,DGoal).
tq(smallest(V,Goal), smallest(Vars, DVars, DV, DGoal)) :-
	!,
	variables_in(Goal, Vars),
	copy_term((Vars,V,Goal),(DVars,DV,Goal1)),
	tq(Goal1,DGoal).
tq(highest(V,Goal), highest(Vars, DVars, DV, DGoal)) :-
	!,
	variables_in(Goal, Vars),
	copy_term((Vars,V,Goal),(DVars,DV,Goal1)),
	tq(Goal1,DGoal).
tq(most(I,V,Goal), most(Vars, DVars, DI, DV, DGoal)) :-
	!,
	variables_in(Goal, Vars),
	copy_term((Vars,I,V,Goal),(DVars,DI,DV,Goal1)),
	tq(Goal1,DGoal).
tq(fewest(I,V,Goal), fewest(Vars, DVars, DI, DV, DGoal)) :-
	!,
	variables_in(Goal, Vars),
	copy_term((Vars,I,V,Goal),(DVars,DI,DV,Goal1)),
	tq(Goal1,DGoal).
tq(Goal,TGoal) :-
	functor(Goal,F,N),
	functor(TGoal,F,N),
	tq_args(N,Goal,TGoal).

tq_args(N,Goal,TGoal) :-
	( N =:= 0 ->
	     true
	; arg(N,Goal,GArg),
	  arg(N,TGoal,TArg),
	  tq(GArg,TArg),
	  N1 is N - 1,
	  tq_args(N1,Goal,TGoal)
	).

variables_in(A, Vs) :- variables_in(A, [], Vs).
	
variables_in(A, V0, V) :-
	var(A), !, add_var(V0, A, V).
variables_in(A, V0, V) :-
	ground(A), !, V = V0. 
variables_in(Term, V0, V) :-
	functor(Term, _, N),
	variables_in_args(N, Term, V0, V).

variables_in_args(N, Term, V0, V) :-
	( N =:= 0 ->
	      V = V0
	; arg(N, Term, Arg),
	  variables_in(Arg, V0, V1),
	  N1 is N-1,
	  variables_in_args(N1, Term, V1, V)
	).

add_var(Vs0, V, Vs) :-
	( contains_var(V, Vs0) ->
	      Vs = Vs0
	; Vs = [V|Vs0]
	).


contains_var(Variable, Term) :-
	\+ free_of_var(Variable, Term).

%   free_of_var(+Variable, +Term)
%   is true when the given Term contains no sub-term identical to the
%   given Variable (which may actually be any term, not just a var).
%   For variables, this is precisely the "occurs check" which is
%   needed for sound unification.

free_of_var(Variable, Term) :-
	Term == Variable,
	!,
	fail.
free_of_var(Variable, Term) :-
	compound(Term),
	!,
	functor(Term, _, Arity),
	free_of_var(Arity, Term, Variable).
free_of_var(_, _).

free_of_var(1, Term, Variable) :- !,
	arg(1, Term, Argument),
	free_of_var(Variable, Argument).
free_of_var(N, Term, Variable) :-
	arg(N, Term, Argument),
	free_of_var(Variable, Argument),
	M is N-1, !,
	free_of_var(M, Term, Variable).

%---------------------------------------------------------------------------
answer(Var, Goal) :- findall(Name,(Goal),Answers).

sum(V, Goal, X) :-
	findall(V, Goal, Vs),
	sumlist(Vs, 0, X).

more(X, Y) :-
	X > Y.
%---------------------------------

divide(X,Y, X/Y).
multiply(X,Y,X*Y).
add(X,Y,X+Y).
subtract(X,Y,X-Y).

sumlist([], Sum, Sum).
sumlist([V|Vs], Sum0, Sum) :-
	Sum1 is Sum0 + V,
	sumlist(Vs, Sum1, Sum).

const(V, V).

num(N, N).

largest(Vars, DVars, DV, DGoal) :-
	largest(DV, DGoal),!,
	Vars = DVars.

smallest(Vars, DVars, DV, DGoal) :-
	smallest(DV, DGoal),!,
	Vars = DVars.

most(Vars, DVars, DI, DV, DGoal) :-
	most(DI, DV, DGoal),!,
	Vars = DVars.

fewest(Vars, DVars, DI, DV, DGoal) :-
	fewest(DI, DV, DGoal),!,
	Vars = DVars.

most(Index,Var,Goal) :-
	setof(Index-Var, Goal, Solutions),
	keysort(Solutions, Collect),
	maximum_run(Collect, Index).

maximum_run(Solutions, Index) :-
	maximum_run(Solutions, foo, 0, Index).

maximum_run([], Index, _Count, Index) :- !.
maximum_run([Index1-_|Rest], BestIndex0, Count0, BestIndex) :-
	first_run(Rest, Index1, 1, Count1, Rest1),
	( Count1 > Count0 ->
	     BestIndex2 = Index1,
	     Count2 = Count1
	; BestIndex2 = BestIndex0,
	  Count2 = Count0
	),
	maximum_run(Rest1, BestIndex2, Count2, BestIndex).

first_run([], _Index, N, N, []).
first_run([Index-G|Rest0], Target, N0, N, Rest) :-
	( Target = Index ->
	     N1 is N0 + 1,
	     first_run(Rest0, Target, N1, N, Rest)
	; N = N0,
	  Rest = [Index-G|Rest0]
	).

fewest(Index,Var,Goal) :-
	setof(Index-Var, Goal, Solutions),
	keysort(Solutions, Collect),
	minimum_run(Collect, Index).



