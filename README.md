# DUAL USE IN CHEM: exploring ways to censor chemical data to mitigate dual use risks.
As we explore strageties to mitigate dual use risks in predictive chemistry (DURPC), we present our data-level mitigation strategy: Selective Noise Addition. 
In pursuit of public distribution of chemical data in safe ways, we test adding noise to only selected data in the dataset identified as sensitive. 
We test this method with three models: 

1. 1-D Polynomial Regression
2. Multilayer Perceptron (MLP)
3. Graph Convolutional Network (GCN) predicting lipophilicity

Read the [paper](https://arxiv.org/abs/2304.10510)

```bibtex
@article{campbell2023censoring,
      title={Censoring chemical data to mitigate dual use risk}, 
      author={Quintina L. Campbell and Jonathan Herington and Andrew D. White},
      year={2023},
      eprint={2304.10510},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
