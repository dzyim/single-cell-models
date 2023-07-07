# class `PertData`

<br>

## Object Attributes

- `data_path`: Path
- `default_GO_graph`: bool (default: True) **OR** `default_pert_graph`: bool (default: True)
- `gene_set_path`: Path (default: None)
- `dataset_name`: str
- `dataset_path`: Path
- `adata`: anndata.AnnData (self.load() -> scanpy.read_h5ad())
- `dataset_processed`: dict (self.load() -> self.create_dataset_file() -> self.create_cell_graph_dataset())
- `ctrl_adata`: anndata.AnnData
- `gene_names`: list
- `node_map`: dict
- `split`: str
- `seed`: int
- `train_gene_set_size`: float
- `subgroup`: dict (self.prepare_split() -> DataSplitter().split_data())
- `gene2go`: dict[str, set] (self.\_\_init\_\_() --> pickle.load())
- `pert_names`: numpy.ndarray (self.load() -> self.set_pert_genes())
- `node_map_pert`: dict (self.load() -> self.set_pert_genes())
 
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
