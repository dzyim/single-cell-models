# The architecture

<br>

## `model.py`

- class **TransformerModel**(nn.Module)

	- self.encoder [class **GeneEncoder**(nn.Module)]
	- self.value_encoder [class **ContinuousValueEncoder**(nn.Module)] OR self.value_encoder [class **CategoryValueEncoder**(nn.Module)] OR self.value_encoder [nn.Identity]
	- self.batch_encoder [class **BatchLabelEncoder**(nn.Module)]
	- self.dsbn [class **DomainSpecificBatchNorm1d**(nn.Module)] OR self.bn [nn.BatchNorm1d]
	- self.transformer_encoder [nn.TransformerEncoder]

	- self.decoder [class **ExprDecoder**(nn.Module)]
		- `mlm_output`, `mlm_zero_probs`
	- self.cls_decoder [class **ClsDecoder**(nn.Module)]
		- `cls_output`
	- self.mvc_decoder [class **MVCDecoder**(nn.Module)]
		- `mvc_output`, `mvc_zero_probs`
	- self.grad_reverse_discriminator [class **AdversarialDiscriminator**(nn.Module)]
		- `dab_output`
	- self.sim [class **Similarity**(nn.Module)]
		- `loss_cce`, `loss_ecs`

<br>

## `generation_model.py`

- class **TransformerGenerator**(nn.Module)

