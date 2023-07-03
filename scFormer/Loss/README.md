# Training

<br>

## Notes

- A run-time error:
    - In this line (https://github.com/bowang-lab/scFormer/blob/2df344a75e5f57cdfa296987786e040912d33935/examples/perturbation/dev_perturb.py#L228): `x.shape` should be `(batch_size * n_genes, 2)`.
    - But when running **scFormer/examples/perturbation/dev_perturb.py**, the actual `x.shape` is `(batch_size * n_genes, 1)`.
    - So an error occurs in this line (https://github.com/bowang-lab/scFormer/blob/2df344a75e5f57cdfa296987786e040912d33935/examples/perturbation/dev_perturb.py#L230).
        - "IndexError: index 1 is out of bounds for dimension 1 with size 1"

<br>

## Q & A

1. **在 scFormer/examples/perturbation/dev_perturb.py 模型训练中，存在哪些损失函数？**
- `loss_mse`: *masked MSE loss* between `mlm_output` and `target_values` （表达量水平，默认计算）
- `loss_cls`: *cross entropy loss* between `cls_output` and `target_labels` （细胞类型，默认**不**计算此loss）
- `loss_cce`: *contrastive cell embedding objective*: *cross entropy loss* between ? （默认**不**计算此loss）
- `loss_mvc`: *masked MSE loss* between `mvc_output` and `target_values` （表达量水平，默认**不**计算此loss）
- `loss_ecs`: *elastic cell similarity objective* （弹性细胞相似性，默认**不**计算此loss）

<br>

2. **如何计算 masked MSE loss 和 elastic cell similarity loss ？**

- Elastic Cell Similarity loss:  ![ECS loss](https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2022/11/22/2022.11.20.517285/embed/graphic-10.gif)
<br>

```python
import torch
import torch.nn.functional as F

def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()

def elastic_cell_similarity_loss(
    cell_emb: torch.Tensor, esc_threshold: float = 0.85
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

3. **scFormer/examples/perturbation/dev_perturb.py 的损失函数计算，涉及哪些超参数？**
- `MLM` (default: True)  # whether to use masked language modeling, currently it is always on.
- `CLS` (default: False)  # celltype classification objective
- `CCE` (default: False)  # Contrastive cell embedding objective
- `MVC` (default: False)  # Masked value prediction for cell embedding
- `ECS` (default: False)  # Elastic cell similarity objective
- `cell_emb_style` (default: "cls")
- `mvc_decoder_style` (default: "inner product, detach")
- `ecs_threshold` (default: 0.85)
- `cce_weight` (default: 10)  # In the script, it's a magic value without a variable name
- `ecs_weight` (default: 10)  # In the script, it's a magic value without a variable name

<br>

4. **scFormer/examples/perturbation/dev_perturb.py 如何评估 perturbation 预测结果？**

- Pearson 相关系数: ![](https://www.biorxiv.org/content/biorxiv/early/2022/11/22/2022.11.20.517285/T4.medium.gif)

<br>

