# 1. Batch Correction

<br>

# 2. Perturbation Response Prediction


## Datasets

- **Dixit et al, 2016.** [[Article]](https://www.cell.com/cell/fulltext/S0092-8674(16)31610-5) [[Harvard Dataverse]](https://dataverse.harvard.edu/api/access/datafile/6154416) [[Lab]](https://www.broadinstitute.org/regev-lab)
- **Adamson et al, 2016.** [[Article]](https://www.cell.com/cell/fulltext/S0092-8674(16)31660-9) [[Harvard Dataverse]](https://dataverse.harvard.edu/api/access/datafile/6154417) [[Lab]](https://weissman.wi.mit.edu/)
- **Norman et al, 2019.** [[Article]](https://www.science.org/doi/10.1126/science.aax4438) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/601096) [[Harvard Dataverse]](https://dataverse.harvard.edu/api/access/datafile/6154020) [[Lab]](https://weissman.wi.mit.edu/)
- **Papalexi et al, 2021.** [[Article]](https://www.nature.com/articles/s41588-021-00778-2) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2020.06.28.175596) [[Lab]](https://satijalab.org/) [[Vignette]](https://satijalab.org/seurat/articles/mixscape_vignette.html)

<br>

## scGPT Example Code

### Runtime error in [`scFormer/examples/perturbation/dev_perturb.py`](https://github.com/bowang-lab/scFormer/blob/main/examples/perturbation/dev_perturb.py)

![](https://raw.githubusercontent.com/dzyim/single-cell-models/main/scGPT/Examples/figures/dev_perturb.py.err.png)

- In [this line](https://github.com/bowang-lab/scFormer/blob/2df344a75e5f57cdfa296987786e040912d33935/examples/perturbation/dev_perturb.py#L228): `x.shape` should be `(batch_size * n_genes, 2)`. But when running scFormer/examples/perturbation/dev_perturb.py, the actual `x.shape` is `(batch_size * n_genes, 1)`.
- An error occurs in [this line](https://github.com/bowang-lab/scFormer/blob/2df344a75e5f57cdfa296987786e040912d33935/examples/perturbation/dev_perturb.py#L230).
  - `IndexError: index 1 is out of bounds for dimension 1 with size 1`

<br>

### Runtime error in [`scGPT/tutorials/Tutorial_Perturbation.ipynb`](https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Perturbation.ipynb)

![](https://raw.githubusercontent.com/dzyim/single-cell-models/main/scGPT/Examples/figures/tutorial_perturbation.py.err.png)

- An error occurs in this line: `model.load_state_dict(model_dict)`.
  - `RuntimeError: Error(s) in loading state_dict for TransformerGenerator:
        Unexpected key(s) in state_dict: "transformer_encoder.layers.0.self_attn.Wqkv.weight",
"transformer_encoder.layers.0.self_attn.Wqkv.bias", "transformer_encoder.layers.1.self_attn.Wqkv.weight",
"transformer_encoder.layers.1.self_attn.Wqkv.bias", "transformer_encoder.layers.2.self_attn.Wqkv.weight",
"transformer_encoder.layers.2.self_attn.Wqkv.bias", "transformer_encoder.layers.3.self_attn.Wqkv.weight",
"transformer_encoder.layers.3.self_attn.Wqkv.bias", "transformer_encoder.layers.4.self_attn.Wqkv.weight",
"transformer_encoder.layers.4.self_attn.Wqkv.bias", "transformer_encoder.layers.5.self_attn.Wqkv.weight",
"transformer_encoder.layers.5.self_attn.Wqkv.bias", "transformer_encoder.layers.6.self_attn.Wqkv.weight",
"transformer_encoder.layers.6.self_attn.Wqkv.bias", "transformer_encoder.layers.7.self_attn.Wqkv.weight",
"transformer_encoder.layers.7.self_attn.Wqkv.bias", "transformer_encoder.layers.8.self_attn.Wqkv.weight",
"transformer_encoder.layers.8.self_attn.Wqkv.bias", "transformer_encoder.layers.9.self_attn.Wqkv.weight",
"transformer_encoder.layers.9.self_attn.Wqkv.bias", "transformer_encoder.layers.10.self_attn.Wqkv.weight",
"transformer_encoder.layers.10.self_attn.Wqkv.bias", "transformer_encoder.layers.11.self_attn.Wqkv.weight",
"transformer_encoder.layers.11.self_attn.Wqkv.bias".`

<br>

## New Implementation

- [Project Report of First Try](https://api.wandb.ai/links/dzyim/sk2jamte)

<br>

### DataModule

- [superclass] [class **gears.PertData**](https://github.com/dzyim/single-cell-models/tree/main/GEARS/PertData)

- object methods:
  - [x] \_\_init\_\_()
  - [x] preprocess()
  - [x] set_vocab()
  - [x] reload_dataset()
  - [x] get_gene_ids()
  - [x] get_pert_flags()
  - [x] get_tokenized_batch()

  - [ ] create_dataset() [**Overriding**]
  - [ ] get_dataloader() [**Overriding**]

<br>

### PertModel

- Modified from `scgpt.model.TransformerModel`:

  - `self.pert_encoder = nn.Embedding(3, d_model, padding_idx=pert_pad_id)`
  - `total_embeddings = src + values + perts`

<br>

### PertTrainer

- object methods:
  - [x] \_\_init\_\_()
  - [x] prepare_data()
  - [x] prepare_model()
  - [x] prepare_batch()
  - [x] fit_epoch()
  - [x] evaluate_epoch()
  - [x] fit()
  
  - [x] predict_batch()
  - [ ] plot_perturbation()
  - [x] eval_testdata()

<br>
