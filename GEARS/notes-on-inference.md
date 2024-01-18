# inference.py 
<br>

## 函数 `non_dropout_analysis`

函数的返回结果是 `pert_metric` (a dict)，它收集了针对每一个扰动(pert)类别的多种性能评估指标。这些指标旨在评估模型在预测非失活(non-dropout)基因表达方面的准确性和可靠性。以下是该字典可能包含内容的细节：

- `frac_correct_direction_top20_non_dropout`：对于扰动类型pert中的前20个非失活基因，预测值与控制ctrl比较后，方向性与真实值一致的比例。
- `frac_opposite_direction_top20_non_dropout`：对于扰动类型pert中的前20个非失活基因，预测值与控制ctrl比较后，方向性完全相反的比例。
- `frac_0/1_direction_top20_non_dropout`：对于扰动类型pert中的前20个非失活基因，预测值与控制ctrl比较后，方向性不明确的比例。（这说明预测的方向性改变不是完全相反）

对于每个非零的基因(non_zero_idx)和非失活的基因(non_dropout_gene_idx)，这些计算也会进行，生成以下指标：

- `frac_correct_direction_non_zero`：预测值与控制ctrl比较后，方向性与真实值一致的非零基因的比例。
- `frac_opposite_direction_non_zero`：预测值与控制ctrl比较后，方向性与真实值完全相反的非零基因的比例。
- `frac_0/1_direction_non_zero`：预测值与控制ctrl比较后，方向性不明确的非零基因的比例。

相同的比例计算也会针对非失活基因 (non_dropout_gene_idx)，添加到 `pert_metric` 中，例如`frac_correct_direction_non_dropout`等。

此外，还进行了以下分析：

- `frac_in_range_non_dropout`：在非零、非失活基因中，预测值在真实值范围内的比例。
- `frac_in_range_45_55_non_dropout`：在非零、非失活基因中，预测值在真实值的 45%-55% 量化区间内的比例。
- `frac_in_range_40_60_non_dropout`、`frac_in_range_25_75_non_dropout`：在更宽的量化区间内，预测值的准确度。
- `mean_sigma_non_dropout`、`std_sigma_non_dropout`、`frac_sigma_below_1_non_dropout`、`frac_sigma_below_2_non_dropout`：这些指标与预测值与真实值之间的标准偏差有关，计算模型预测的偏离程度。

最后，还会计算与差异表达的前20个非失活基因相关的皮尔森相关系数和均方误差，这些值相对于控制条件以及它们自身的值进行比较。

每个扰动 (pert) 的指标都被添加到 `pert_metric` 字典的对应项下，这样就为每种扰动提供了一组详细的性能评估。这些分析对于理解模型在不同实验条件下的预测行为非常重要，有助于识别模型的优势和改进的领域。

## 函数 `deeper_analysis`

函数 `deeper_analysis` 的结果为 `pert_metric` 字典，包含以下内容：

- `frac_correct_direction_all`：表示对于所有基因而言，模型预测的方向性变化是否与实际变化相匹配的百分比。

- `frac_correct_direction_xxx`：针对不同数量的差异表达基因（比如 top 20, 50, 100, 200），表示模型预测方向性正确的百分比。

- `frac_in_range` 和其他类似的项（如 `frac_in_range_45_55`, `frac_in_range_40_60`, `frac_in_range_25_75`）：所有非零表达的 top 20 差异表达基因中预测值与实际值匹配的范围内的百分比。

- `mean_sigma`, `std_sigma`, `frac_sigma_below_1`, `frac_sigma_below_2`: 描述模型预测值与实际值之间方差的度量。

- `fold_change_gap_all` 和特定的折叠变化间隙指标（如 `fold_change_gap_downreg_0.33`, `fold_change_gap_upreg_3`, `fold_change_gap_upreg_10`）：折叠变化间隙比对指标，反映模型预测的折叠变化与实际数据的折叠变化之间的差异。

- 以 `_delta` 后缀标记的指标：表示模型预测的变化值（以控制条件为基准）与真实数据之间的皮尔森相关性以及均方误差。

适用于 top200_hvg（最变异的200个基因）、top20_de, top50_de, top100_de, top200_de 等扰动种类的相关度量，表示这些特定基因集合的相关性和误差指标。

在分析模型的性能时，这些指标非常重要。它们不仅反映了模型在有限的、生物学上重要的基因集上的表现，还提供了关于模型预测准确性、方向性判定能力和预测折叠变化的额外信息。通过这些详细的指标，研究人员可以深入理解模型在特定生物学上下文中的性能和潜在的缺陷，为后续的调优和生物学洞见提供支持。

<br>

[内容由 GPT 生成]

<br>
