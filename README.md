# ContextGNNBERT
Work in progress code of our paper ContextGNNBERT

To run training and report evaluation on the train, validation and test set, execute:

```
python3 main.py \
	--dataset_name super_glue:boolq \
	--cols_for_context passage question \
	--context_masker TFIDFContextMasker \
	--context_mask_fn_kwargs '{"percentile_cutoff": [75, 50]}' \
	--truncation_strategy only_first \
	--num_train_epochs 50 \
	--eval_steps 100 \
	--batch_size 512
```

or to run the script without wandb experiment tracking, execute as:
```
WANDB_MODE=disabled python3 main.py \
	--dataset_name scitail:snli_format \
	--label_col gold_label \
	--pos_label entailment \
	--cols_for_context sentence1 sentence2 \
	--context_masker TFIDFContextMasker \
	--context_mask_fn_kwargs '{"percentile_cutoff": 50}' \
	--truncation_strategy longest_first \
	--num_train_epochs 50 \
	--eval_steps 100 \
	--batch_size 512
```