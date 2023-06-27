# 损失函数

<br>

## Q&A

1. **在 scGPT/examples/finetune_integration.py 模型训练中，存在哪些损失函数？**
- `loss_mse`: masked MSE loss between `mlm_output` and `target_values` （表达量水平）
- `loss_zero_log_prob`: negative log-likelihood loss between `mlm_zero_probs` and `target_values > 0` （表达量是否为零）
- `loss_gepc`: masked MSE loss between `mvc_output` and `target_values` （表达量水平）
- `loss_gepc_zero_log_prob`: negative log-likelihood loss between `mvc_zero_probs` and `target_values > 0` （表达量是否为零）
- `loss_ecs`: elastic cell similarity loss （ ？）
- `loss_dab`: cross entropy loss between `dab_output` and `batch_labels` （批次标签）

<br>

2. **如何计算 masked MSE loss 和 negative log-likelihood loss ？**
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
```

<br>

3. **scGPT/examples/finetune_integration.py 的损失函数计算，涉及哪些超参数？**
- `GEPC` (default: True)
- `explicit_zero_prob` (default: True)
- `ecs_thres` (default: 0.8)
- `dab_weight` (default: 1.0)

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
