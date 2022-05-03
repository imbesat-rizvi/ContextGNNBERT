# ContextGNNBERT

To run training and report evaluation on the train, validation and test set, execute:

```
python3 main.py \
	--dataset_name scitail:snli_format \
	--label_col gold_label \
	--pos_label entailment \
	--cols_for_context sentence1 sentence2 \
	--context_masker TFIDFContextMasker \
	--context_mask_fn_kwargs '{"percentile_cutoff": 50}' \
	--truncation_strategy longest_first \
	--classifier_net GATv2Conv \
	--num_train_epochs 50 \
	--batch_size 128 \
	--optimizer_name AdamW \
	--optimizer_kwargs '{"lr": 1e-5, "eps": 1e-8}'
```

or to run the script without wandb experiment tracking, execute as:
```
WANDB_MODE=disabled python3 main.py ...
```