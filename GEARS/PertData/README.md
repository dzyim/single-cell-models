# class `PertData`

<br>

## Object Attributes

- `data_path`: `Path | str`
- `default_GO_graph`: `bool` (default: True) **OR** `default_pert_graph`: `bool` (default: True)
- `gene_set_path`: `Optional[Path | str]` (default: None)
- `dataset_name`: `str`
- `dataset_path`: `Path | str`
- `adata`: `anndata.AnnData` (self.load() -> scanpy.read_h5ad())
- `dataset_processed`: `dict[str, list[torch_geometric.data.Data]]` (self.load() -> self.create_dataset_file() -> self.create_cell_graph_dataset())
- `ctrl_adata`: `anndata.AnnData`
- `gene_names`: `pandas.Series`
- `node_map`: `dict[str, int]`
- `split`: `str` (default: 'simulation')
- `seed`: `int` (default: 1)
- `train_gene_set_size`: `float` (default: 0.75)
- `subgroup`: `dict[str, dict[str, list[str]]]` (self.prepare_split() -> DataSplitter().split_data())
- `gene2go`: `dict[str, set[str]]` (self.\_\_init\_\_() --> pickle.load())
- `pert_names`: `numpy.ndarray[U10]` (self.load() -> self.set_pert_genes())
- `node_map_pert`: `dict[str, int]` (self.load() -> self.set_pert_genes())
 
<br>

## Methods

- `set_pert_genes()`
- `load()`
- `new_data_process()`
- `prepare_split()`
- `get_dataloader()`
- `create_dataset_file()`
- `get_pert_idx()`
- `create_cell_graph()`
- `create_cell_graph_dataset()`

<br>
