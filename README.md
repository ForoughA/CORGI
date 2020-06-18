# CORGI

This is the code for the [Conversational Neuro-Symbolic Commonsense Reasoning](https://arxiv.org/abs/2006.10022).

## Dependencies
1. PyTorch 
2. Spacy	
3. higgingface en_coref_lg model: https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_lg-3.0.0/en_coref_lg-3.0.0.tar.gz
4. python2
5. numpy
6. py
7. rpython

## Navigation
1. logging/
	folder to store user interaction logs under `<USER_NAME>` exaplained below.
2. testnet/
	contains the code for our neuro-symbolic theorem prover built on top of [spyrolog](https://github.com/leonweber/spyrolog)
3. data/
	contains our proposed commonsense reasoning benchmark. Look at data/REAME.md for a data description.
4. net.py, attention.py, config.py, modeltest.py
	are files relevant to the inference for our neuro-symbolic theorem prover
5. TypeDict.json
	modified dictionary of types built on top of Aristo tuple KB v1.03 Mar 2017 Release
6. prolog_info-reasoning-17.pkl, model_testing-reasoning-17.tch
	pretrained models for the neuro-symbolic theorem prover
7. facts7.txt, functor_arity7.txt
	knowledge base of commonsense facts
8. user-study-data.txt 
	statements used in the user study

## Running the code:

in order to run the code use prompt:
```
python2 reasoning.py --user <USER_NAME> --engine /testnet
```

additional arguments are: 
* `--verbose true` (can be used for debugging purposes to see more printouts)
* `--resume <STATEMENT_NUMBER>` (can be used to resume the study at a certain statement if needed)

