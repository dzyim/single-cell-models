# Single Cell Models

> 单细胞测序的分析模型汇总和学习笔记

> Model zoo and study notes for single cell data analysis.

<br>

## Basics

### Books
- **Single-cell best practices** (by [Theislab](https://www.github.com/theislab)) https://www.sc-best-practices.org/
- **Analysis of single cell RNA-seq data** https://www.singlecellcourse.org/
- **Orchestrating Single-Cell Analysis with Bioconductor** https://bioconductor.org/books/release/OSCA/
- **Best Practices for Spatial Transcriptomics Analysis with Bioconductor** https://lmweber.org/BestPracticesST/

### Papers
- **Best practices for single-cell analysis across modalities (2023)** [[paper]](https://www.nature.com/articles/s41576-023-00586-w)
- **Current best practices in single‐cell RNA‐seq analysis: a tutorial (2019)** [[paper]](https://www.embopress.org/doi/full/10.15252/msb.20188746) 

<br>

## Datasets

- **Human Cell Atlas** [[Home]](https://www.humancellatlas.org/) [[Data Portal]](https://data.humancellatlas.org/) [[paper]](https://elifesciences.org/articles/27041) [[pdf]](https://cdn.elifesciences.org/articles/27041/elife-27041-v2.pdf)
  - **Human Developmental Cell Atlas** [[UK Team]](https://www.hdbr.org/) [[Sweden Team]](https://hdca-sweden.scilifelab.se/) [[Roadmap]](https://www.nature.com/articles/s41586-021-03620-1)

- **CZ CELLxGENE** [[Home]](https://cellxgene.cziscience.com/) [[CellGuide]](https://cellxgene.cziscience.com/cellguide) [[Census package]](https://chanzuckerberg.github.io/cellxgene-census/index.html) [[Census R package]](https://chanzuckerberg.github.io/cellxgene-census/r/index.html)
  - **Tabula Sapiens** [[Home]](https://tabula-sapiens-portal.ds.czbiohub.org/) [[Data]](https://tabula-sapiens-portal.ds.czbiohub.org/whereisthedata) [[paper]](https://www.science.org/stoken/author-tokens/ST-495/full) [[pdf]](https://www.science.org/doi/pdf/10.1126/science.abl4896)

<br>

## Perturb-seq Datasets

- **Dixit et al, 2016.** [[Article]](https://www.cell.com/cell/fulltext/S0092-8674(16)31610-5) [[GEO]](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063) [[Harvard Dataverse]](https://dataverse.harvard.edu/api/access/datafile/6154416) [[Regev Lab]](https://www.broadinstitute.org/regev-lab) [[pdf]](https://www.cell.com/action/showPdf?pii=S0092-8674%2816%2931610-5)

- **Adamson et al, 2016.** [[Article]](https://www.cell.com/cell/fulltext/S0092-8674(16)31660-9) [[GEO]](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546) [[Harvard Dataverse]](https://dataverse.harvard.edu/api/access/datafile/6154417) [[Weissman Lab]](https://weissman.wi.mit.edu/) [[pdf]](https://www.cell.com/action/showPdf?pii=S0092-8674%2816%2931660-9)
  - **Direction of perturbations:** CRISPRi (gene knockdown)
  - **Number of perturbed genes:** one-gene perturbations
  - **Cell type:** K562

- **Norman et al, 2019.** [[Article]](https://www.science.org/doi/10.1126/science.aax4438) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/601096) [[GEO]](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) [[Harvard Dataverse]](https://dataverse.harvard.edu/api/access/datafile/6154020) [[Weissman Lab]](https://weissman.wi.mit.edu/) [[pdf]](https://www.science.org/doi/pdf/10.1126/science.aax4438?download=false)
  - **Direction of perturbations:** CRISPRa (gene activation)
  - **Number of perturbed genes:** one-gene perturbations and two-gene perturbations
  - **Cell type:** K562

- **Papalexi et al, 2021.** [[Article]](https://www.nature.com/articles/s41588-021-00778-2) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2020.06.28.175596) [[GEO]](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153056) [[Satija Lab]](https://satijalab.org/) [[Vignette]](https://satijalab.org/seurat/articles/mixscape_vignette.html)

- **Replogle et al, 2022.** [[Article]](https://www.cell.com/cell/fulltext/S0092-8674(22)00597-9) [[Data Portal]](https://gwps.wi.mit.edu/) [[GitHub1]](https://github.com/josephreplogle/guide_calling) [[GitHub2]](https://github.com/thomasmaxwellnorman/perturbseq_demo) [[Weissman Lab]](https://weissman.wi.mit.edu/) [[pdf]](https://www.cell.com/action/showPdf?pii=S0092-8674%2822%2900597-9)

<br>

### Collection
- **scPerturb.org** Single Cell Perturbation Datasets [[Home]](http://projects.sanderlab.org/scperturb/) [[GitHub]](https://github.com/sanderlab/scPerturb) [[Sander Lab]](https://www.sanderlab.org/)

<br>

## Early Methods

- **limma (2003).** [[Home]](https://bioinf.wehi.edu.au/limma/) [[Article]](https://www.sciencedirect.com/science/article/abs/pii/S1046202303001555) [[Bioconductor]](https://www.bioconductor.org/packages/release/bioc/html/limma.html) Linear Models for Microarray and RNA-seq Data
- **ComBat (2007).** [[Article1]](https://academic.oup.com/biostatistics/article/8/1/118/252073) [[Article2]](https://academic.oup.com/nargab/article/2/3/lqaa078/5909519) [[GitHub]](https://github.com/zhangyuqing/ComBat-seq) removing known batch effects using empirical Bayes frameworks
- **svaseq (2014).** [[Article]](https://academic.oup.com/nar/article/42/21/e161/2903156) [[Bioconductor]](https://bioconductor.org/packages/release/bioc/html/sva.html) [[GitHub]](https://github.com/jtleek/sva-devel) removing batch effects with known control probes
- **Seurat v2 (2018).** [[Home]](https://satijalab.org/seurat/) [[Article]](https://www.nature.com/articles/nbt.4096) [[GitHub]](https://github.com/satijalab/seurat) [[Lab]](https://satijalab.org/) Tools for Single Cell Genomics
- **fastMNN (2018).** [[Article]](https://www.nature.com/articles/nbt.4091) [[Bioconductor]](https://bioconductor.org/packages/release/bioc/html/batchelor.html) [[GitHub]](https://github.com/LTLA/batchelor/) [[Lab]](https://www.mdc-berlin.de/haghverdi) batch effect correction by matching mutual nearest neighbors
- **MOFA (2018).** [[Home]](https://biofam.github.io/MOFA2/) [[Article]](https://www.embopress.org/doi/full/10.15252/msb.20178124) [[GitHub]](https://github.com/bioFAM/MOFA) [[pdf]](https://www.embopress.org/doi/pdf/10.15252/msb.20178124) Multi-Omics Factor Analysis
- **LIGER (2019).** [[Article]](https://www.cell.com/cell/fulltext/S0092-8674(19)30504-5) [[CRAN]](https://cran.r-project.org/web/packages/rliger/) [[GitHub]](https://github.com/welch-lab/liger) [[PyLiger]](https://github.com/welch-lab/pyliger) [[Lab]](https://welch-lab.github.io/) Linked Inference of Genomic Experimental Relationships
- **Harmony (2019).** [[Home]](https://portals.broadinstitute.org/harmony/index.html) [[Article]](https://www.nature.com/articles/s41592-019-0619-0) [[CRAN]](https://cran.r-project.org/web/packages/harmony/) [[GitHub]](https://github.com/immunogenomics/harmony) [[harmonypy]](https://github.com/slowkow/harmonypy) [[Lab]](https://immunogenomics.hms.harvard.edu/) Fast, sensitive and accurate integration of single-cell data
- **Scanorama (2019).** [[Home]](https://cb.csail.mit.edu/cb/scanorama/) [[Article]](https://www.nature.com/articles/s41587-019-0113-3) [[GitHub]](https://github.com/brianhie/scanorama) [[Lab]](https://people.csail.mit.edu/bab/) Panoramic stitching of single cell data
- **GeoSketch (2019).** [[Home]](https://cb.csail.mit.edu/cb/geosketch/) [[Article]](https://www.cell.com/cell-systems/fulltext/S2405-4712(19)30152-8) [[GitHub]](https://github.com/brianhie/geosketch) [[Lab]](https://people.csail.mit.edu/bab/) Geometric sketching compactly summarizes the single-cell transcriptomic landscape
- **MOFA+ (2020).** [[Home]](https://biofam.github.io/MOFA2/) [[Article]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02015-1) [[Bioconductor]](https://www.bioconductor.org/packages/release/bioc/html/MOFA2.html) [[GitHub]](https://github.com/bioFAM/MOFA2) [[pdf]](https://genomebiology.biomedcentral.com/counter/pdf/10.1186/s13059-020-02015-1.pdf) Multi-Omics Factor Analysis v2
- **MEFISTO (2020).** [[Home]](https://biofam.github.io/MOFA2/) [[Article]](https://www.nature.com/articles/s41592-021-01343-9) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2020.11.03.366674) [[GitHub]](https://github.com/bioFAM/mofapy2) [[pdf]](https://www.nature.com/articles/s41592-021-01343-9.pdf) a flexible and versatile toolbox for modeling high-dimensional data when spatial or temporal dependencies between the samples are known
- **Cellij (2023).** [[Article]](https://icml-compbio.github.io/2023/papers/WCBICML2023_paper124.pdf) [[GitHub]](https://github.com/bioFAM/cellij) [[pdf]](https://icml-compbio.github.io/2023/papers/WCBICML2023_paper124.pdf) A Modular Factor Model Framework

<br>

## Integrated Methods

- **SCENIC (2017).** [[Home]](https://scenic.aertslab.org/) [[GitHub]](https://github.com/aertslab/SCENIC) [[Lab]](https://aertslab.org/) [[pySCENIC]](https://github.com/aertslab/pySCENIC) [[Tutorials]](https://scenic.aertslab.org/tutorials/) single-cell regulatory network inference and clustering
  - GENIE3. [[GitHub]](https://github.com/aertslab/GENIE3) [[Bioconductor]](https://bioconductor.org/packages/devel/bioc/html/GENIE3.html) [[Vignette]](https://bioconductor.org/packages/release/bioc/vignettes/GENIE3/inst/doc/GENIE3.html)
  - GRNBoost. [[GitHub]](https://github.com/aertslab/GRNBoost)
  - RcisTarget. [[GitHub]](https://github.com/aertslab/RcisTarget) [[Bioconductor]](https://bioconductor.org/packages/release/bioc/html/RcisTarget.html) [[Vignette]](https://bioconductor.org/packages/release/bioc/vignettes/RcisTarget/inst/doc/RcisTarget_MainTutorial.html)
  - AUCell. [[GitHub]](https://github.com/aertslab/AUCell) [[Bioconductor]](https://bioconductor.org/packages/release/bioc/html/AUCell.html) [[Vignette]](https://bioconductor.org/packages/release/bioc/vignettes/AUCell/inst/doc/AUCell.html)

- **SCENIC+ (2023).** [[GitHub]](https://github.com/aertslab/scenicplus) [[Lab]](https://aertslab.org/) [[Tutorials]](https://scenicplus.readthedocs.io/en/latest/tutorials.html) single-cell multiomic inference of enhancers and gene regulatory networks
  - LoomXpy. [[GitHub]](https://github.com/aertslab/LoomXpy)
  - pycisTopic. [[GitHub]](https://github.com/aertslab/pycisTopic)
  - pycistarget. [[GitHub]](https://github.com/aertslab/pycistarget)
  - create_cisTarget_databases. [[GitHub]](https://github.com/aertslab/create_cisTarget_databases)
  <!--  - pySCENIC. [[GitHub]](https://github.com/aertslab/pySCENIC) --->

- **scArches (2020).** [[Article]](https://www.nature.com/articles/s41587-021-01001-7) [[Docs]](https://docs.scarches.org/en/latest/about.html) [[GitHub]](https://github.com/theislab/scarches) [[Theis Lab]](https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab) [[pdf]](https://www.nature.com/articles/s41587-021-01001-7.pdf) Single-cell architecture surgery, a package for reference-based analysis of single-cell data
  - scVI (Lopez et al., 2018)
  - trVAE (Lotfollahi et al., 2020)
  - scANVI (Xu et al., 2019)
  - scGen (Lotfollahi et al., 2019)
  - expiMap (Lotfollahi et al., 2023)
  - totalVI (Gayoso al., 2019)
  - treeArches (Michielsen et al., 2022)
  - SageNet (Heidari et al., 2022)
  - mvTCR (Drost et al., 2022)
  - scPoli (De Donno et al., 2022)

- **Mowgli (2023).** [[Article]](https://www.nature.com/articles/s41467-023-43019-2) [[Doc]](https://mowgli.readthedocs.io/en/latest/) [[GitHub]](https://github.com/cantinilab/mowgli) [[pdf]](https://www.nature.com/articles/s41467-023-43019-2.pdf) Multi Omics Wasserstein inteGrative anaLysIs

<br>

## Convolutional Neural Networks

- **scBasset (2021).** [[Article]](https://www.nature.com/articles/s41592-022-01562-8) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2021.09.08.459495) [[GitHub]](https://github.com/calico/scBasset) [[Calico Labs]](https://www.calicolabs.com/) Sequence-based modeling of single cell ATAC-seq using convolutional neural networks

<br>

## Variational Autoencoders

- **scVI (2018).** [[Home]](https://scverse.org/) [[Article]](https://www.nature.com/articles/s41592-018-0229-2) [[GitHub]](https://github.com/scverse/scvi-tools) [[Guide]](https://docs.scvi-tools.org/en/stable/user_guide/index.html) [[Yosef Lab]](https://yoseflab.github.io/) Single-cell variational inference tools
- **scGen (2018).** [[Article]](https://www.nature.com/articles/s41592-019-0494-8) [[Docs]](https://scgen.readthedocs.io/en/stable/) [[GitHub]](https://github.com/theislab/scgen) [[Theis Lab]](https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab) [[pdf]](https://rdcu.be/bMlbD) Single cell perturbation prediction
- **trVAE (2019).** [[Article]](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i610/6055927) [[GitHub]](https://github.com/theislab/trVAE) [[Theis Lab]](https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab) Conditional out-of-distribution prediction using transfer VAE
- **scANVI (2019).** [[Home]](https://scvi-tools.org/) [[Article]](https://www.embopress.org/doi/full/10.15252/msb.20209620) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/532895) [[GitHub]](https://github.com/scverse/scvi-tools/blob/main/scvi/model/_scanvi.py) [[Yosef Lab]](https://yoseflab.github.io/) single-cell ANnotation using Variational Inference
- **VEGA (2020).** [[Article]](https://www.nature.com/articles/s41467-021-26017-0) [[Docs]](https://vega-documentation.readthedocs.io/en/latest/index.html) [[GitHub]](https://github.com/LucasESBS/vega) VAE Enhanced by Gene Annotations
- **totalVI (2020).** [[Home]](https://scvi-tools.org/) [[Article]](https://www.nature.com/articles/s41592-020-01050-x) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2020.05.08.083337) [[GitHub]](https://github.com/scverse/scvi-tools/blob/main/scvi/model/_totalvi.py) [[Yosef Lab]](https://yoseflab.github.io/) Total Variational Inference
- **PeakVI (2021).** [[Home]](https://scvi-tools.org/) [[Article]](https://www.sciencedirect.com/science/article/pii/S2667237522000376) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2021.04.29.442020) [[GitHub]](https://github.com/scverse/scvi-tools/blob/main/scvi/model/_peakvi.py) [[Yosef Lab]](https://yoseflab.github.io/) [[pdf]](https://www.sciencedirect.com/science/article/pii/S2667237522000376/pdf) A Deep Generative Model for Single Cell Chromatin Accessibility Analysis
- **MultiVI (2021).** [[Home]](https://scvi-tools.org/) [[Article]](https://www.nature.com/articles/s41592-023-01909-9) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2021.08.20.457057) [[GitHub]](https://github.com/scverse/scvi-tools/blob/main/scvi/model/_multivi.py) [[Yosef Lab]](https://yoseflab.github.io/) [[pdf]](https://www.nature.com/articles/s41592-023-01909-9.pdf) A deep generative model for the integration of multimodal data
- **CPA (2021).** [[Article]](https://www.embopress.org/doi/full/10.15252/msb.202211517) [[GitHub1]](https://github.com/facebookresearch/CPA) [[GitHub2]](https://github.com/theislab/cpa) [[Theis Lab]](https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab) [[pdf]](https://www.embopress.org/doi/pdfdirect/10.15252/msb.202211517) The Compositional Perturbation Autoencoder learns effects of perturbations at the single-cell level
- **chemCPA (2022).** [[Article]](https://neurips.cc/virtual/2022/poster/53227) [[GitHub]](https://github.com/theislab/chemcpa) [[Theis Lab]](https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab) [[pdf]](https://openreview.net/pdf?id=vRrFVHxFiXJ) Predicting Cellular Responses to Novel Drug Perturbations at a Single-Cell Resolution
- **MrVI (2022).** [[BioRxiv]](https://www.biorxiv.org/content/early/2022/10/06/2022.10.04.510898) [[GitHub]](https://github.com/YosefLab/mrvi) [[Yosef Lab]](https://yoseflab.github.io/) [[pdf]](https://www.biorxiv.org/content/early/2022/10/06/2022.10.04.510898.full.pdf) Multi-resolution Variational Inference for quantifying sample-level heterogeneity in single-cell omics

<br>

## Graph Neural Network Models

- **scGCN (2020).** [[Article]](https://www.nature.com/articles/s41467-021-24172-y) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2020.09.13.295535) [[GitHub]](https://github.com/QSong-github/scGCN) [[pdf]](https://www.nature.com/articles/s41467-021-24172-y.pdf) a graph convolutional networks algorithm for knowledge transfer in single cell omics
- **GLUE (2021).** [[Article]](https://www.nature.com/articles/s41587-022-01284-4) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2021.08.22.457275) [[GitHub]](https://github.com/gao-lab/GLUE) [[pdf]](https://www.nature.com/articles/s41587-022-01284-4.pdf) Graph-Linked Unified Embedding
- **DeepMAPS (2021).** [[Article]](https://www.nature.com/articles/s41467-023-36559-0) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2021.10.31.466658) [[GitHub]](https://github.com/OSU-BMBL/deepmaps) [[pdf]](https://www.nature.com/articles/s41467-023-36559-0.pdf) Single-cell biological network inference using a heterogeneous graph transformer
- **GEARS (2022).** [[Article]](https://www.nature.com/articles/s41587-023-01905-6) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2022.07.12.499735) [[GitHub]](https://github.com/snap-stanford/GEARS) [[Leskovec Lab]](http://snap.stanford.edu/) [[pdf]](https://www.nature.com/articles/s41587-023-01905-6.pdf) a geometric deep learning model that predicts outcomes of novel multi-gene perturbations

<br>

## (Pretrained) Foundation Models

- **scBERT (2021).** [[Article]](https://www.nature.com/articles/s42256-022-00534-z) [[GitHub]](https://github.com/TencentAILabHealthcare/scBERT) [[Lab]](https://ai.tencent.com/ailab) [[pdf]](https://www.nature.com/articles/s42256-022-00534-z.pdf) a large-scale pretrained deep language model for cell type annotation
- **tGPT (2022).** [[Article]](https://www.sciencedirect.com/science/article/pii/S2589004223006132) [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2022.01.31.478596) [[HuggingFace]](https://huggingface.co/lixiangchun/transcriptome-gpt-1024-8-16-64) [[Lab]](https://www.tmucih.com/) [[pdf]](https://www.cell.com/iscience/pdf/S2589-0042(23)00613-2.pdf) Generative pretraining from large-scale transcriptomes for single-cell deciphering
- **Geneformer (2022).** [[Article]](https://www.nature.com/articles/s41586-023-06139-9) [[Data]](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M) [[HuggingFace]](https://huggingface.co/ctheodoris/Geneformer) [[Lab]](https://www.ellinorlab.org/) [[pdf]](https://cqb.pku.edu.cn/zenglab_cn/pdf/s41586-023-06139-9.pdf) Transfer learning enables predictions in network biology
- **scFormer (2022).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2022.11.20.517285) [[GitHub]](https://github.com/bowang-lab/scFormer) [[Lab]](https://wanglab.ml/) [[pdf]](https://www.biorxiv.org/content/10.1101/2022.11.20.517285.full.pdf) a universal representation learning approach for single-cell data using transformers
- **SATURN (2023).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.02.03.526939) [[GitHub]](https://github.com/snap-stanford/SATURN) [[Leskovec Lab]](http://snap.stanford.edu/) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.02.03.526939.full.pdf) Towards Universal Cell Embeddings: Integrating Single-cell RNA-seq Datasets across Species with SATURN
- **xTrimoGene (2023).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.03.24.534055) [[Lab]](https://www.biomap.com/) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.03.24.534055.full.pdf) an efficient and scalable representation learner for single-cell RNA-seq data
- **scGPT (2023).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.04.30.538439) [[GitHub]](https://github.com/bowang-lab/scGPT) [[Lab]](https://wanglab.ml/) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.04.30.538439.full.pdf) towards building a foundation model for single-cell multi-omics using generative AI
- **scFoundation (2023).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.05.29.542705) [[GitHub]](https://github.com/biomap-research/scFoundation) [[Lab]](https://www.biomap.com/) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.05.29.542705.full.pdf) large scale foundation model on single-cell transcriptomics
- **GET (2023).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.09.24.559168) [[GitHub]](https://github.com/GET-Foundation) [[HuggingFace]](https://huggingface.co/get-foundation) [[Lab]](https://sailing-lab.github.io/) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.09.24.559168.full.pdf) a foundation model of transcription across human cell types
- **CellPolaris (2023).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.09.25.559244) [[GitHub]](https://github.com/xCompass-AI/CellPolaris) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.09.25.559244.full.pdf) Decoding Cell Fate through Generalization Transfer Learning of Gene Regulatory Networks
- **GeneCompass (2023).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.09.26.559542) [[GitHub]](https://github.com/xCompass-AI/GeneCompass) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.09.26.559542.full.pdf) Deciphering Universal Gene Regulatory Mechanisms with Knowledge-Informed Cross-Species Foundation Model
- **CellPLM (2023).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.10.03.560734) [[GitHub]](https://github.com/OmicsML/CellPLM) [[Lab]](https://dse.cse.msu.edu/) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.10.03.560734.full.pdf) Pre-training of Cell Language Model Beyond Single Cells
- **GenePT (2023).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.10.16.562533) [[GitHub]](https://github.com/yiqunchen/GenePT) [[Lab]](https://www.james-zou.com/) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.10.16.562533.full.pdf) GenePT: A Simple But Hard-to-Beat Foundation Model for Genes and Cells Built from ChatGPT
- **UCE (2023).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.11.28.568918) [[GitHub]](https://github.com/snap-stanford/UCE) [[Leskovec Lab]](http://snap.stanford.edu/) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.11.28.568918.full.pdf) Universal Cell Embeddings: A Foundation Model for Cell Biology

<br>

## Benchmarks

- **A benchmark of batch-effect correction methods for single-cell RNA sequencing data. (2020)** [[Article]](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1850-9) [[GitHub]](https://github.com/JinmiaoChenLab/Batch-effect-removal-benchmarking) [[pdf]](https://genomebiology.biomedcentral.com/counter/pdf/10.1186/s13059-019-1850-9.pdf)
- **Benchmarking atlas-level data integration in single-cell genomics. (2022)** [[Article]](https://www.nature.com/articles/s41592-021-01336-8) [[GitHub]](https://github.com/theislab/scib-reproducibility) [[pipeline]](https://github.com/theislab/scib-pipeline) [[pdf]](https://www.nature.com/articles/s41592-021-01336-8.pdf)
- **Microsoft/zero-shot-scfoundation** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.10.16.561085) [[GitHub]](https://github.com/microsoft/zero-shot-scfoundation) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.10.16.561085.full.pdf)

<br>

## Related Lists

- [List of awesome-deep-learning-single-cell-papers by OmicsML](https://github.com/OmicsML/awesome-deep-learning-single-cell-papers)
- [List of awesome-foundation-model-single-cell-papers by OmicsML](https://github.com/OmicsML/awesome-foundation-model-single-cell-papers)
- [Survey of single cell models by Qi Liu](https://github.com/qiliu-ghddi/singlecell_gpt?tab=readme-ov-file#survey-andor-references)

<br>

## Appendix: Technologies
<br>

<img src="https://www.singlecellcourse.org/figures/moores-law.png" width="80%">

Figure 1: Scaling of scRNA-seq experiments (image from [Svensson et al.](https://arxiv.org/pdf/1704.01379.pdf))

<br>
