### DUAL USE IN CHEM: exploring ways to censor chemical data to mitigate dual use risks.
As we explore strageties to mitigate dual use risks in predictive chemistry (DURPC), we present our data-level mitigation strategy: Selective Noise Addition. 
In pursuit of public distribution of chemical data in safe ways, we test adding noise to only selected data in the dataset with labels identified as sensitive. 
We test this method with three models: 

1. 1-D Polynomial Regression
2. Multilayer Perceptron (MLP)
3. Graph Convolutional Network (GCN) predicting lipophilicity

Read the [paper](https://arxiv.org/abs/2304.10510)

## Setup

```bash
conda env create -f environment.yml
conda activate dualusage
python -m ipykernel install --user --name dualusage  # may not be needed
```

## Usage

**1-D Polynomial Regression**

Open and run `SimplePolynomial.ipynb` in the root directory.

**MLP — quick comparison across censoring types**

Open and run `mlp_task/mlp_quick_comparison_run.ipynb`.

**MLP — full sweep**

```bash
cd mlp_task
python papermill_run.py
```

This tests all censoring types across intensity levels. Executed notebooks are written to `mlp_task/OUTPUTS/notebooks/` and the results figure notebook to `mlp_task/OUTPUTS/mlp_main_plots.ipynb`.

**GCN — full sweep**

```bash
cd gcn_task
python papermill_run.py
```

This tests all censoring types across intensity levels. Executed notebooks are written to `gcn_task/OUTPUTS/notebooks/` and postprocessing notebooks to `gcn_task/OUTPUTS/postprocess_stuff/`.

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
