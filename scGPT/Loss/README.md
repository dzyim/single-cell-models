# 损失函数

<br>

## Q&A

1. 在 scGPT/examples/finetune_integration.py 模型训练中，存在哪些损失函数？
- `loss_mse`: masked MSE loss between `mlm_output` and `target_values` （表达量水平）
- `loss_zero_log_prob`: negative log-likelihood loss between `mlm_zero_probs` and `target_values > 0` （表达量是否为零）
- `loss_gepc`: masked MSE loss between `mvc_output` and `target_values` （表达量水平）
- `loss_gepc_zero_log_prob`: negative log-likelihood loss between `mvc_zero_probs` and `target_values > 0` （表达量是否为零）
- `loss_ecs`: elastic cell similarity loss （？）
- `loss_dab`: cross entropy loss between `dab_output` and `batch_labels` （批次标签）

<br>

2. scGPT/examples/finetune_integration.py 的损失函数计算，涉及哪些超参数？
- `GEPC` (default: True)
- `explicit_zero_prob` (default: True)
- `ecs_thres` (default: 0.8)
- `dab_weight` (default: 1.0)

<br>

3. `mlm` 和 `mvc` 等分别是什么意思？
- MLM: Masked Language Modelling
- MGM: Masked Gene Modelling
- MVC: Masked Value Prediction for Cell Embedding
- ECS: Elastic Cell Similarity
- DAB: Domain Adaptation by Reverse Backpropagation
- DAR: Domain Adaptation by Reverse Backpropagation

<br>

4. scGPT 的 Transformer 模型有哪些 Task heads？
- ![scFormer Fig. 1](https://www.biorxiv.org/content/biorxiv/early/2022/11/22/2022.11.20.517285/F1.large.jpg)

