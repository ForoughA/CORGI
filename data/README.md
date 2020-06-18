## Commonsense Reasoning Benchmark Dataset

* The data is presented in "tab delimited text" format. 
* The first line are column headers and there are 6 columns available: "statements", "Template", "Annotation1", "Annotation2", "Annotation3", "Annotation4".
* "Statements" indicates the if-then-because command collected from humans.
* "Template" refers to the logic template available in Table 5 in the appendix.
* "Annotationi" for i\in[1,4] indicates the commonsense presumption annotations. Annotations are tuples of (index, missing_text) where index is the word index of the place to insert missing_text in the original statement.

if you use this dataset in your research please cite our paper with the following BibTeX entry:

```
@article{arabshahi2020conversational,
  title={Conversational Neuro-Symbolic Commonsense Reasoning},
  author={Arabshahi, Forough and Lee, Jennifer and Gawarecki, Mikayla and Mazaitis, Kathryn and Azaria, Amos and Mitchell, Tom},
  journal={arXiv preprint arXiv:2006.10022},
  year={2020}
}
```
