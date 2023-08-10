import sys
import pickle
import warnings
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import anndata as ad
import numpy as np
import requests
from deprecation import deprecated
from scipy import sparse
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
from tqdm import tqdm

from .data_utils import get_DE_genes, get_dropout_non_zero_genes
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer.gene_tokenizer import GeneVocab, tokenize_and_pad_batch

warnings.filterwarnings("ignore")

def print2(x):
    """Print the value to the stream stderr."""
    print(x, file=sys.stderr, flush=True)

class PertDataset(Dataset):
    """
    A PyTorch Dataset subclass for storing a dataset of expression
    values and metadata (e.g. perturbation conditions) of gene or 
    compound perturbations.

    This class is designed to handle different perturbation types, including
    gene perturbations and compound perturbations. It assumes that the input
    control and perturbed data are properly aligned, ensuring that each item
    in the dataset contains information about a specific perturbation pair.
    
    Args:
        x: An AnnData object containing the expression data of control cells.
        y: An AnnData object containing the expression data of perturbed cells.

    Attributes:
        x: An AnnData object containing the expression data of control cells.
        y: An AnnData object containing the expression data of perturbed cells.
        shape: The shape of the expression data (x and y with the same shape).
        var: The metadata that include gene symbols and gene annotations.
        uns: A dictionary-like structure for storing unstructured annotations.

    Methods:
        __getitem__: Returns the item (a dictionary with a pair of cell-wise
                     expression values and metadata) at the given index.
        __len__: Returns the total number of items in the dataset.

    Example usage:
    ```
    # Create a PertDataset instance
    indices = np.random.randint(0, len(ctrl_adata), len(adata))
    dataset = PertDataset(x=ctrl_adata[indices], y=adata)

    # Access an item from the dataset
    item = dataset[0]
    ```
    """

    def __init__(self, x: ad.AnnData, y: ad.AnnData) -> None:
        assert x.shape == y.shape
        self.x = x
        self.y = y
        self.shape = y.shape
        self.var = y.var.copy()
        self.uns = y.uns.copy()

    def __getitem__(self, idx: int) -> dict[str, dict[str, np.ndarray|dict]]:
        x_ = self.x[idx]
        y_ = self.y[idx]
        return {
            'x': {
                'X': x_.X.A.reshape(x_.n_vars),
                'obs': x_.obs.reset_index().to_dict('records')[0]
            },
            'y': {
                'X': y_.X.A.reshape(y_.n_vars),
                'obs': y_.obs.reset_index().to_dict('records')[0]
            },
        }

    def __len__(self) -> int:
        return self.shape[0]

class PertData:
    """A customized version of gears.PertData with a subset of methods."""

    def __init__(self, data_path, 
                 gene_set_path=None, 
                 default_pert_graph=True) -> None:
        # Dataset/Dataloader attributes
        self.data_path = Path(data_path)
        self.default_pert_graph: bool = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}

        # Split attributes
        self.split = None
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None

        if not self.data_path.exists():
            self.data_path.mkdir(parents=True)

    @staticmethod
    def dataverse_download(url, save_path) -> None:
        """A dataverse download helper with progress bar.
    
        Args:
            url (str): the url of the dataset
            path (str): the path to save the dataset
        """
        if Path(save_path).exists():
            print2('Found local copy...')
        else:
            print2("Downloading...")
            response = requests.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(save_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()

    def zip_data_download_wrapper(self, url, save_path) -> None:
        if Path(save_path).exists():
            print2('Found local copy...')
        else:
            file_path = str(save_path) + '.zip'
            PertData.dataverse_download(url, file_path)
            print2('Extracting zip file...')
            with ZipFile(file_path, 'r') as f:
                f.extractall(path=self.data_path)
            print2("Done!")
            
    def load(self, data_name=None, data_path=None) -> None:
        """Load existing dataset.

        Args:
            data_name: for loading norman/adamson/dixit datasets (default: None)
            data_path: for loading local datasets via a path (default: None)
        """
        if data_name in ['norman', 'adamson', 'dixit']:
            ## load from harvard dataverse
            URL = {
                'norman':  'https://dataverse.harvard.edu/api/access/datafile/6154020',
                'adamson': 'https://dataverse.harvard.edu/api/access/datafile/6154417',
                'dixit':   'https://dataverse.harvard.edu/api/access/datafile/6154416',
            }
            data_path = self.data_path / data_name
            self.zip_data_download_wrapper(URL[data_name], data_path)
        elif data_name and (self.data_path / data_name).exists():
            data_path = self.data_path / data_name
        elif Path(data_path).exists():
            data_path = Path(data_path)
        else:
            raise ValueError("data attribute is either Norman/Adamson/Dixit "
                             "or a path with a perturb_processed.h5ad file.")
        self.adata = ad.read_h5ad(data_path / 'perturb_processed.h5ad')
        self.dataset_name = data_path.name
        self.dataset_path = data_path
        #self.ctrl_adata = self.adata[self.adata.obs["condition"] == "ctrl"]
        #self.gene_names = self.adata.var["gene_name"].tolist()

        pyg_path = Path(data_path) / 'data_pyg'
        if not pyg_path.exists():
            pyg_path.mkdir()
        
class PerturbData(PertData):

    def __init__(self,
                 data_path: Path|str,
                 mode: str = "gene",
                 test_size: float = 0.1,
                 ctrl_flag: int = 0,
                 crispr_flag: int = 1,
                 cls_token: str = "<cls>",
                 pad_token: str = "<pad>",
                 pad_value: int = -2,
                 pert_pad_id: int = 2,
                 special_tokens: tuple[str] = ("<pad>", "<cls>", "<eoc>"),
                 **kwargs) -> None:
        super().__init__(data_path, **kwargs)
        self.mode: str = mode
        if self.mode == "gene":
            self.gene_col = "gene_name"
            self.cond_col = "condition"
            self.ctrl_str = "ctrl"
            self.pert_col = "condition"
        elif self.mode == "compound":
            self.gene_col = "symbol"
            self.cond_col = "treatment"
            self.ctrl_str = "S0000"
            self.pert_col = "canonical_smiles"
        else:
            raise ValueError("Perturbation mode should be 'gene' or 'compound'!")
        self.test_size = float(test_size)
        self.ctrl_flag = int(ctrl_flag)
        self.crispr_flag = int(crispr_flag)
        self.cls_token: str = cls_token
        self.pad_token: str = pad_token
        self.pad_value = int(pad_value)
        self.pert_pad_id = int(pert_pad_id)
        self.special_tokens = special_tokens
        self.vocab = None
        self.dataset = None

    def preprocess(self, hvg: int|bool = False) -> None:
        if self.adata is None:
            raise AttributeError('adata not loaded!')
        if len(self.adata.layers) and "counts" in self.adata.layers:
            preprocessor = Preprocessor(
                use_key = "counts",
                filter_gene_by_counts = 3,       # Step 1
                filter_cell_by_counts = False,   # Step 2
                normalize_total = 1e4,           # Step 3: whether to normalize the raw data and to what sum
                result_normed_key = "X_normed",  # the key in adata.layers to store the normalized data
                log1p = True,                    # Step 4: whether to log1p the normalized data  # log1p = data_is_raw
                result_log1p_key = "X_log1p",    # the key in adata.layers to store the log transformed data
                subset_hvg = hvg,                # Step 5: whether to subset the raw data to highly variable genes
                hvg_flavor = "seurat_v3",        # hvg_flavor = "seurat_v3" if data_is_raw else "cell_ranger"
                binning = 51,                    # Step 6: whether to bin the raw data and to what number of bins
                result_binned_key = "X_binned",  # the key in adata.layers to store the binned data
            )
            preprocessor(self.adata, batch_key=None)
            self.adata.X = sparse.csr_matrix(self.adata.layers['X_binned'])
        else:
            preprocessor = Preprocessor(
                use_key = "X",
                filter_gene_by_counts = 3,       # Step 1
                filter_cell_by_counts = False,   # Step 2
                normalize_total = False,         # Step 3: whether to normalize the raw data and to what sum
                result_normed_key = "X_normed",  # the key in adata.layers to store the normalized data
                log1p = False,                   # Step 4: whether to log1p the normalized data  # log1p = data_is_raw
                result_log1p_key = "X_log1p",    # the key in adata.layers to store the log transformed data
                subset_hvg = hvg,                # Step 5: whether to subset the raw data to highly variable genes
                hvg_flavor = "cell_ranger",      # hvg_flavor = "seurat_v3" if data_is_raw else "cell_ranger"
                binning = False,                 # Step 6: whether to bin the raw data and to what number of bins
                result_binned_key = "X_binned",  # the key in adata.layers to store the binned data
            )
            preprocessor(self.adata, batch_key=None)

        self.adata = get_dropout_non_zero_genes(
            get_DE_genes(self.adata, skip_calc_de=False)
        )

    @deprecated(details='Use self.create_dataset() instead.')
    def reload_dataset(self) -> None:
        if self.dataset_path:
            dataset_file = Path(self.dataset_path) / 'data_pyg' / 'cell_graphs.pkl'
        else:
            raise AttributeError('dataset_path not found!')
        # Note: Need self.ctrl_adata for self.create_dataset_file()!
        self.ctrl_adata = self.adata[self.adata.obs[self.cond_col] == self.ctrl_str]
        self.gene_names = self.adata.var[self.gene_col]
        print2("Creating pyg object for each cell in the data...")
        self.dataset_processed = self.create_dataset_file()
        print2("Saving new dataset pyg object at " + str(dataset_file))
        with open(dataset_file, 'wb') as f:
            pickle.dump(self.dataset_processed, f)
        print2("Done!")

    def set_vocab(self, vocab_file: Optional[Path] = None) -> None:
        if vocab_file:
            vocab = GeneVocab.from_file(vocab_file)
            for s in self.special_tokens:
                if s not in vocab:
                    vocab.append_token(s)
        elif self.adata is not None:
            genes = self.adata.var[self.gene_col].tolist()
            vocab = Vocab(
                VocabPybind(list(genes) + list(self.special_tokens), None)
            ) # bidirectional lookup [gene <-> int]
        else:
            raise AttributeError('vocab_file not given and adata not loaded!')
        vocab.set_default_index(vocab[self.pad_token])
        self.vocab = vocab
    
    def get_gene_ids(self) -> np.ndarray[int]:
        if self.vocab is None:
            raise AttributeError('vocab not set!')
        if self.adata is None:
            raise AttributeError('adata not loaded!')
        genes = self.adata.var[self.gene_col].tolist()
        return np.array(self.vocab(genes), dtype=int)
    
    def get_pert_flags(self, condition: str) -> Optional[np.ndarray[int]]:
        gene_ids = self.get_gene_ids()
        pert_ids = [self.vocab[g] for g in condition.split('+') if g != self.ctrl_str]
        #TODO: make pert_ids always a subset of gene_ids
        if not len(pert_ids):
            return np.full(len(gene_ids), self.pert_pad_id, dtype=int)
        elif np.setdiff1d(pert_ids, gene_ids).size:
            return None
        else:
            return np.where(
                np.isin(gene_ids, pert_ids), self.crispr_flag, self.ctrl_flag
            )

    def get_tokenized_batch(
            self, batch_data, append_cls: bool = False,
            include_zero_gene: str|bool = True, max_len: int = 0
        ) -> dict:
        #TODO: allow include_zero_gene = False
        batch_size: int = batch_data['y']['X'].shape[0]
        n_genes: int = batch_data['y']['X'].shape[1]
        if not max_len > 0:
            max_len: int = n_genes + 1 if append_cls else n_genes
        if self.mode == "compound":
            pert = {'pert': batch_data['y']['obs'][self.pert_col]}
            input_idx = list(range(batch_size))
        elif self.mode == "gene":
            input_idx = []
            pert_list = []
            for idx, cond in enumerate(batch_data['y']['obs'][self.pert_col]):
                flags = self.get_pert_flags(cond)
                if flags is not None:
                    pert_list.append(flags)
                    input_idx.append(idx)
            try:
                pert_array = np.stack(pert_list, axis=0)
            except ValueError:
                return {}
            pert = tokenize_and_pad_batch(
                pert_array,
                self.get_gene_ids(),
                max_len=max_len,
                vocab=self.vocab,
                cls_token=self.cls_token,
                pad_token=self.pad_token,
                pad_value=self.pad_value,
                append_cls=append_cls,
                include_zero_gene=include_zero_gene,
            )
            pert['pert_flags'] = pert.pop('values', None)
        else:
            raise ValueError("Perturbation mode should be 'gene' or 'compound'!")
        x = tokenize_and_pad_batch(
            batch_data['x']['X'].view(
                batch_size, n_genes).detach().cpu().numpy()[input_idx],
            self.get_gene_ids(),
            max_len=max_len,
            vocab=self.vocab,
            cls_token=self.cls_token,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=append_cls,
            include_zero_gene=include_zero_gene,
        )
        y = tokenize_and_pad_batch(
            batch_data['y']['X'].view(
                batch_size, n_genes).detach().cpu().numpy()[input_idx],
            self.get_gene_ids(),
            max_len=max_len,
            vocab=self.vocab,
            cls_token=self.cls_token,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=append_cls,
            include_zero_gene=include_zero_gene,
        )
        y['target_values'] = y.pop('values', None)
        #de = {'de_idx': [batch_data.de_idx[i] for i in input_idx]}
        return pert|x|y

    def create_dataset(self) -> None:
        self.ctrl_adata = self.adata[self.adata.obs[self.cond_col] == self.ctrl_str]
        indices = np.random.randint(0, len(self.ctrl_adata), len(self.adata))
        self.dataset = PertDataset(x=self.ctrl_adata[indices, ], y=self.adata)

    def get_dataloader(
            self, batch_size: int, test_batch_size: Optional[int] = None
        ) -> None:
        if self.dataset is None:
            raise AttributeError('dataset not created!')
        if test_batch_size is None:
            test_batch_size = batch_size
        train_x, test_x, train_y, test_y = train_test_split(
            self.dataset.x, self.dataset.y, test_size=self.test_size, shuffle=True
        )
        train_x, valid_x, train_y, valid_y = train_test_split(
            train_x, train_y, test_size=self.test_size, shuffle=True
        )
        train_loader = DataLoader(
            PertDataset(train_x, train_y),
            batch_size=batch_size, shuffle=True, drop_last = True
        )
        val_loader = DataLoader(
            PertDataset(valid_x, valid_y),
            batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            PertDataset(test_x, test_y),
            batch_size=batch_size, shuffle=False
        )
        self.dataloader = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
        }
