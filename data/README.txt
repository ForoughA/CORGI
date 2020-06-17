Commonsense Reasoning Benchmark Dataset is available in this folder.
The data is presented in "tab delimited text" format. 
The first line are column headers and there are 6 columns available: "statements", "Template", "Annotation1", "Annotation2", "Annotation3", "Annotation4".
"Statements" indicates the if-then-because command collected from humans.
"Template" refers to the logic template available in Table 5 in the appendix.
"Annotationi" for i\in[1,4] indicates the commonsense presumption annotations. Annotations are tuples of (index, missing_text) where index is the word index of the place to insert missing_text in the original statement.