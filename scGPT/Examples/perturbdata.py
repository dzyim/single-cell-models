import sys
import pickle
from pathlib import Path

import torch
import numpy as np
from scipy import sparse
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

from gears import PertData
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer.gene_tokenizer import GeneVocab, tokenize_and_pad_batch


def print2(x):
    '''
    Print the value to the stream stderr.
    '''
    print(x, file=sys.stderr, flush=True)


class PerturbData(PertData):

    def __init__(self, data_path,
                 ctrl_flag: int = 1, knockout_flag: int = 0,
                 cls_token: str = "<cls>",
                 pad_token: str = "<pad>", pad_value: int = -2,
                 special_tokens: tuple[str] = ("<pad>", "<cls>", "<eoc>"),
                 **kwargs) -> None:
        super().__init__(data_path, **kwargs)
        self.ctrl_flag = ctrl_flag
        self.knockout_flag = knockout_flag
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.pad_value = pad_value
        self.special_tokens = special_tokens
        self.vocab = None

    def preprocess(self, hvg: int|bool = False) -> None:
        if self.adata is None:
            raise AttributeError('adata not loaded!')
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

    def reload_dataset(self) -> None:
        if self.dataset_path:
            dataset_fname = Path(self.dataset_path) / 'data_pyg' / 'cell_graphs.pkl'
        else:
            raise AttributeError('dataset_path not found!')
        # Note: Need self.ctrl_adata for self.create_dataset_file()!
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        print2("Creating pyg object for each cell in the data...")
        self.dataset_processed = self.create_dataset_file()
        print2("Saving new dataset pyg object at " + str(dataset_fname))
        with open(dataset_fname, 'wb') as f:
            pickle.dump(self.dataset_processed, f)
        print2("Done!")

    def set_vocab(self, vocab_file: Path|str|None = None) -> None:
        if vocab_file:
            vocab = GeneVocab.from_file(vocab_file)
            for s in self.special_tokens:
                if s not in vocab:
                    vocab.append_token(s)
        elif self.adata is not None:
            genes = self.adata.var["gene_name"].tolist()
            vocab = Vocab(
                VocabPybind(genes + self.special_tokens, None)
            )  # bidirectional lookup [gene <-> int]
        else:
            raise AttributeError('vocab_file not given and adata not loaded!')
        vocab.set_default_index(vocab[self.pad_token])
        self.vocab = vocab
    
    def get_gene_ids(self) -> np.ndarray[int]:
        if self.vocab is None:
            raise AttributeError('vocab not set!')
        if self.adata is None:
            raise AttributeError('adata not loaded!')
        genes = self.adata.var["gene_name"].tolist()
        return np.array(self.vocab(genes), dtype=int)
    
    def get_pert_flags(self, condition: str) -> np.ndarray[int]|None:
        gene_ids = self.get_gene_ids()
        pert_ids = [self.vocab[g] for g in condition.split('+') if g != 'ctrl']
        #TODO: make pert_ids always a subset of gene_ids
        if np.setdiff1d(pert_ids, gene_ids).size:
            return None
        else:
            return np.where(
                np.isin(gene_ids, pert_ids), self.knockout_flag, self.ctrl_flag
            )

    def get_tokenized_batch(
            self, batch_data, append_cls: bool = False,
            include_zero_gene: bool = True
        ) -> dict:
        batch_size: int = len(batch_data.y)
        n_genes: int = self.adata.shape[1]
        max_len: int = n_genes + 1 if append_cls else n_genes
        input_idx = []
        pert_list = []
        for idx, cond in enumerate(batch_data.pert):
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
        x = tokenize_and_pad_batch(
            batch_data.x.view(batch_size, n_genes
                ).detach().cpu().numpy()[input_idx],
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
            batch_data.y.detach().cpu().numpy()[input_idx],
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
        return pert|x|y

