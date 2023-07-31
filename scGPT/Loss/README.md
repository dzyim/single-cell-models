# 损失函数

<br>

## Q&A

1. **在 scGPT/examples/finetune_integration.py 模型训练中，存在哪些损失函数？**
- `loss_mse`: *masked MSE loss* between `mlm_output` and `target_values` （表达量水平）
- `loss_zero_log_prob`: *negative log-likelihood loss* between `mlm_zero_probs` and `target_values > 0` （表达量是否为零）
- `loss_gepc`: *masked MSE loss* between `mvc_output` and `target_values` （表达量水平）
- `loss_gepc_zero_log_prob`: *negative log-likelihood loss* between `mvc_zero_probs` and `target_values > 0` （表达量是否为零）
- `loss_ecs`: *elastic cell similarity loss*
- `loss_dab`: *cross entropy loss* between `dab_output` and `batch_labels` （批次标签）

<br>

2. **如何计算 masked MSE loss、negative log-likelihood loss、以及 elastic cell similarity loss ？**
```python
import torch
import torch.nn.functional as F

def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()

def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()

def elastic_cell_similarity_loss(
    cell_emb: torch.Tensor, ecs_threshold: float = 0.8
) -> torch.Tensor:
    # normalize the embedding
    cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
    cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)
    # mask out diagnal elements
    mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
    cos_sim = cos_sim.masked_fill(mask, 0.0)
    # only optimize positive similarities
    cos_sim = F.relu(cos_sim)
    return torch.mean(1 - (cos_sim - ecs_threshold) ** 2)
```

<br>

3. **scGPT/examples/finetune_integration.py 的损失函数计算，涉及哪些超参数？**
- `GEPC` (default: True)
- `explicit_zero_prob` (default: True)
- `ecs_thres` (default: 0.8)
- `dab_weight` (default: 1.0)
- `ecs_weight` (default: 10)  # In the script, it's a magic value without a variable name

<br>

4. **`mlm_output`, `mvc_output`, `dab_output` 等缩写，分别是什么意思？**
- MLM: Masked Language Modelling
- MGM: Masked Gene Modelling
- MVC: Masked Value Prediction for Cell Embedding
- CLS: Celltype Classification
- CCE: Contrastive Cell Embedding
- ECS: Elastic Cell Similarity
- DAB: Domain Adaptation by Reverse Backpropagation
- DAR: Domain Adaptation by Reverse Backpropagation

<br>

5. **scFormer 和 scGPT 的 Transformer 模型有哪些 task heads？**

![scFormer Fig. 1](https://www.biorxiv.org/content/biorxiv/early/2022/11/22/2022.11.20.517285/F1.large.jpg)
- MGM: Masked Gene Modelling
- MVC: Masked Value Prediction for Cell Embedding
- ECS: Elastic Cell Similarity
- DAR: Domain Adaptation by Reverse Backpropagation

<br>

## Fine-tuning for Perturbation Response Prediction

1. Perturbation response prediction 任务（下面称为“任务”）的微调过程，所输入的 Norman et al., Adamson et al., ... 等数据集的基因表达值是如何预处理的？

<br>

2. 任务所用数据集中，以及 gears.PertData 中的差异表达基因（DE genes）是如何计算出的？

<br>

3. 任务所用数据集的“Ctrl”数据和“Perturbed”数据中，大致多少基因表达值为零值？多少基因在Perturbation前后有显著变化？一个基因在“Ctrl”或“Perturbed”组内的表达值的方差有多大？

<br>

