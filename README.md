# Single Cell Models

> 单细胞测序的分析模型汇总和学习笔记

> Model zoo and study notes for single cell data analysis.

<br>

## Books
<br>

- **Orchestrating Single-Cell Analysis with Bioconductor** https://bioconductor.org/books/release/OSCA/
- **Analysis of single cell RNA-seq data** https://www.singlecellcourse.org/
- **Single-cell best practices** https://www.sc-best-practices.org/

<br>

## Datasets
<br>

- **Human Cell Atlas** [[Home]](https://www.humancellatlas.org/) [[Data Portal]](https://data.humancellatlas.org/)
- **CZ CELLxGENE** [[Home]](https://cellxgene.cziscience.com/) [[Census package]](https://chanzuckerberg.github.io/cellxgene-census/index.html) [[Census R package]](https://chanzuckerberg.github.io/cellxgene-census/r/index.html)

<br>

## Early Methods
<br>

- **limma (2003).** [[Home]](https://bioinf.wehi.edu.au/limma/) [[Article]](https://www.sciencedirect.com/science/article/abs/pii/S1046202303001555) [[Bioconductor]](https://www.bioconductor.org/packages/release/bioc/html/limma.html) Linear Models for Microarray and RNA-seq Data
- **ComBat (2007).** [[Article1]](https://academic.oup.com/biostatistics/article/8/1/118/252073) [[Article2]](https://academic.oup.com/nargab/article/2/3/lqaa078/5909519) [[GitHub]](https://github.com/zhangyuqing/ComBat-seq) removing known batch effects using empirical Bayes frameworks
- **svaseq (2014).** [[Article]](https://academic.oup.com/nar/article/42/21/e161/2903156) [[Bioconductor]](https://bioconductor.org/packages/release/bioc/html/sva.html) [[GitHub]](https://github.com/jtleek/sva-devel) removing batch effects with known control probes
- **Seurat v2 (2018).** [[Home]](https://satijalab.org/seurat/) [[Article]](https://www.nature.com/articles/nbt.4096) [[GitHub]](https://github.com/satijalab/seurat) [[Lab]](https://satijalab.org/) Tools for Single Cell Genomics
- **fastMNN (2018).** [[Article]](https://www.nature.com/articles/nbt.4091) [[Bioconductor]](https://bioconductor.org/packages/release/bioc/html/batchelor.html) [[GitHub]](https://github.com/LTLA/batchelor/) [[Lab]](https://www.mdc-berlin.de/haghverdi) batch effect correction by matching mutual nearest neighbors
- **LIGER (2019).** [[Article]](https://www.cell.com/cell/fulltext/S0092-8674(19)30504-5) [[CRAN]](https://cran.r-project.org/web/packages/rliger/) [[GitHub]](https://github.com/welch-lab/liger) [[PyLiger]](https://github.com/welch-lab/pyliger) [[Lab]](https://welch-lab.github.io/) Linked Inference of Genomic Experimental Relationships
- **Harmony (2019).** [[Home]](https://portals.broadinstitute.org/harmony/index.html) [[Article]](https://www.nature.com/articles/s41592-019-0619-0) [[CRAN]](https://cran.r-project.org/web/packages/harmony/) [[GitHub]](https://github.com/immunogenomics/harmony) [[harmonypy]](https://github.com/slowkow/harmonypy) [[Lab]](https://immunogenomics.hms.harvard.edu/) Fast, sensitive and accurate integration of single-cell data
- **Scanorama (2019).** [[Home]](https://cb.csail.mit.edu/cb/scanorama/) [[Article]](https://www.nature.com/articles/s41587-019-0113-3) [[GitHub]](https://github.com/brianhie/scanorama) [[Lab]](https://people.csail.mit.edu/bab/) Panoramic stitching of single cell data
- **GeoSketch (2019).** [[Home]](https://cb.csail.mit.edu/cb/geosketch/) [[Article]](https://www.cell.com/cell-systems/fulltext/S2405-4712(19)30152-8) [[GitHub]](https://github.com/brianhie/geosketch) [[Lab]](https://people.csail.mit.edu/bab/) Geometric sketching compactly summarizes the single-cell transcriptomic landscape

<br>

## Integrated Methods
<br>

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

<br>

## Variational Autoencoders
<br>

- **scVI (2018).** [[Home]](https://scverse.org/) [[Article]](https://www.nature.com/articles/s41592-018-0229-2) [[GitHub]](https://github.com/scverse/scvi-tools) [[Guide]](https://docs.scvi-tools.org/en/stable/user_guide/index.html) [[Theis Lab]](https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab) single-cell variational inference tools
- **scGen (2018).** [[Article]](https://www.nature.com/articles/s41592-019-0494-8) [[Docs]](https://scgen.readthedocs.io/en/stable/) [[GitHub]](https://github.com/theislab/scgen) [[Theis Lab]](https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab) [[pdf]](https://rdcu.be/bMlbD) single cell perturbation prediction
- **trVAE (2019).** [[Article]](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i610/6055927) [[GitHub]](https://github.com/theislab/trVAE) [[Theis Lab]](https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab) Conditional out-of-distribution prediction using transfer VAE
- **VEGA (2020).** [[Article]](https://www.nature.com/articles/s41467-021-26017-0) [[Docs]](https://vega-documentation.readthedocs.io/en/latest/index.html) [[GitHub]](https://github.com/LucasESBS/vega) VAE Enhanced by Gene Annotations
- **totalVI (2021).** [[Home]](https://scvi-tools.org/) [[Article]](https://www.nature.com/articles/s41592-020-01050-x) [[Theis Lab]](https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab) Total Variational Inference
- **scANVI (2021).** [[Home]](https://scvi-tools.org/) [[Article]](https://www.embopress.org/doi/full/10.15252/msb.20209620) [[Theis Lab]](https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab) single-cell ANnotation using Variational Inference
- **CPA (2021).** [[Article]](https://www.embopress.org/doi/full/10.15252/msb.202211517) [[GitHub1]](https://github.com/facebookresearch/CPA) [[GitHub2]](https://github.com/theislab/cpa) [[Theis Lab]](https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab) [[pdf]](https://www.embopress.org/doi/pdfdirect/10.15252/msb.202211517?download=false) The Compositional Perturbation Autoencoder learns effects of perturbations at the single-cell level.

<br>

## Graph Neural Network Models
<br>

- **GEARS (2022).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2022.07.12.499735) [[GitHub]](https://github.com/snap-stanford/GEARS) [[Lab]](http://snap.stanford.edu/) [[pdf]](https://www.biorxiv.org/content/10.1101/2022.07.12.499735.full.pdf) a geometric deep learning model that predicts outcomes of novel multi-gene perturbations

<br>

## (Pretrained) Language Models
<br>

- **scBERT (2021).** &emsp;&emsp;&emsp; [[Article]](https://www.nature.com/articles/s42256-022-00534-z) [[GitHub]](https://github.com/TencentAILabHealthcare/scBERT) [[Lab]](https://ai.tencent.com/ailab) [[pdf]](https://www.nature.com/articles/s42256-022-00534-z.pdf) a large-scale pretrained deep language model for cell type annotation
- **scFormer (2022).** &emsp;&emsp; [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2022.11.20.517285) [[GitHub]](https://github.com/bowang-lab/scFormer) [[Lab]](https://wanglab.ml/) [[pdf]](https://www.biorxiv.org/content/10.1101/2022.11.20.517285.full.pdf) a universal representation learning approach for single-cell data using transformers
- **xTrimoGene (2023).** &emsp;[[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.03.24.534055) [[Lab]](https://www.biomap.com/) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.03.24.534055.full.pdf) an efficient and scalable representation learner for single-cell RNA-seq data
- **scGPT (2023).** &emsp;&emsp;&emsp;&nbsp; [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.04.30.538439) [[GitHub]](https://github.com/bowang-lab/scGPT) [[Lab]](https://wanglab.ml/) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.04.30.538439.full.pdf) towards building a foundation model for single-cell multi-omics using generative AI
- **scFoundation (2023).** [[BioRxiv]](https://www.biorxiv.org/content/10.1101/2023.05.29.542705) [[GitHub]](https://github.com/biomap-research/scFoundation) [[Lab]](https://www.biomap.com/) [[pdf]](https://www.biorxiv.org/content/10.1101/2023.05.29.542705.full.pdf) large scale foundation model on single-cell transcriptomics

<br>
