import numpy as np
from functools import partial
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

from datasets import load_dataset, ClassLabel, concatenate_datasets

# from datasets import DatasetDict

from data_utils.hf_utils import (
    keep_cols,
    hstack_cols,
    get_tokenizer_encoder,
    tokenize_dataset,
)

import ContextMasker
from models import FCNBERT, ContextGNNBERT, ContextAveragedBERT
from models.utils import HFTrainer, compute_aprfbeta, to_labels

import torch
from torch import optim
from transformers import (
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback,
    TrainingArguments,
)

from arguments import parser
import wandb


def main(
    dataset_name="super_glue",
    config_name="boolq",
    label_col=None,
    pos_label=1,
    input_text_cols=["passage", "question"],
    context_corpus_splits="train",
    context_masker="TFIDFContextMasker",
    context_masker_init_kwargs={},
    context_mask_fn_kwargs={"percentile_cutoff": (75, 50)},
    truncation_strategy="only_first",
    encoder_model="bert-base-uncased",
    trainable_encoder=True,
    classifier_net="GATConv",
    gnn_kwargs={},
    dropout=0.2,
    num_layers=3,
    hidden_channels=[256, 64],
    non_linearity="ReLU",
    load_checkpoint="",
    num_train_epochs=50,
    batch_size=128,
    optimizer_name="AdamW",
    optimizer_kwargs={"lr": 1e-3, "eps": 1e-8},
    encoder_optimizer_kwargs={},
    num_warmup_steps=0,
    early_stopping_patience=5,
    early_stopping_threshold=1e-4,
    no_class_weight=False,
):

    dataset = load_dataset(dataset_name, name=config_name)
    # dataset = DatasetDict({k: v.select(range(10)) for k, v in dataset.items()})
    num_labels = None

    if label_col is None:
        for feat_name, feat_class in dataset["train"].features.items():
            if isinstance(feat_class, ClassLabel):
                label_col = feat_name
                num_labels = feat_class.num_classes
                break
        print(f"No label col specified, inferred label col = {label_col}")

    elif dataset["train"].features[label_col].dtype == "string":
        dataset = dataset.class_encode_column(label_col)
        num_labels = dataset["train"].features[label_col].num_classes

    print(f"Num labels = {num_labels}")

    # rename the HF dataset label col to labels as HF pipelines expect this name
    dataset = dataset.rename_column(label_col, "labels")
    label_col = "labels"
    cols_to_exl_in_model_inp = [
        i for i, j in dataset["train"].features.items() if not isinstance(j, ClassLabel)
    ]

    # print an overview of the dataset
    print(dataset)

    if isinstance(input_text_cols, str):
        context_corpus_col = input_text_cols
    else:
        context_corpus_col = "context_corpus"

    tokenizer, encoder = get_tokenizer_encoder(encoder_model=encoder_model)
    max_tokenized_length = encoder.config.max_position_embeddings

    dataset = tokenize_dataset(
        dataset,
        tokenizer=tokenizer,
        tokenizing_fields=input_text_cols,
        truncation=truncation_strategy,
        padding="max_length",
        max_length=max_tokenized_length,
    )

    if classifier_net == "FCN":
        model = FCNBERT.from_pretrained(
            saved_path=load_checkpoint,
            load_strategy="best",
            encoder=encoder,
            num_labels=num_labels,
            trainable_encoder=trainable_encoder,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            dropout=dropout,
            non_linearity=non_linearity,
        )

    else:

        context_corpus = keep_cols(dataset, cols=input_text_cols)

        if isinstance(context_corpus_splits, str):
            context_corpus = context_corpus[context_corpus_splits]
        else:
            context_corpus = concatenate_datasets(
                [context_corpus[i] for i in context_corpus_splits]
            )

        context_corpus = hstack_cols(
            context_corpus,
            cols=input_text_cols,
            stacked_col_name=context_corpus_col,
        )

        context_masker = ContextMasker.__dict__[context_masker](
            context_corpus[context_corpus_col],
            tokenizer=tokenizer,
            **context_masker_init_kwargs,
        )

        dataset = context_masker.insert_context_mask(
            dataset, cols=input_text_cols, **context_mask_fn_kwargs
        )

        if classifier_net == "ContextAveraged":
            model = ContextAveragedBERT.from_pretrained(
                saved_path=load_checkpoint,
                load_strategy="best",
                encoder=encoder,
                num_labels=num_labels,
                trainable_encoder=trainable_encoder,
                num_context_types=(1 if isinstance(input_text_cols, str) else 2),
                num_layers=num_layers,
                hidden_channels=hidden_channels,
                dropout=dropout,
                non_linearity=non_linearity,
            )

        else:
            model = ContextGNNBERT.from_pretrained(
                saved_path=load_checkpoint,
                load_strategy="best",
                encoder=encoder,
                num_labels=num_labels,
                trainable_encoder=trainable_encoder,
                gnn_class=classifier_net,
                gnn_kwargs=gnn_kwargs,
                gnn_block_dropout=dropout,
                num_layers=num_layers,
                hidden_channels=hidden_channels,
                non_linearity=non_linearity,
            )

    print(model)

    dataset = dataset.remove_columns(cols_to_exl_in_model_inp)
    dataset.set_format("torch")

    encoder_training = "tuned_encoder" if trainable_encoder else "untuned_encoder"
    run_name = f"{dataset_name}-{config_name}-{classifier_net}-{encoder_model}-{encoder_training}"
    trainer_dir = Path("trainer_outputs")
    trainer_dir.mkdir(parents=True, exist_ok=True)
    output_dir = trainer_dir / run_name

    train_args = TrainingArguments(
        run_name=run_name,
        report_to="wandb",
        num_train_epochs=num_train_epochs,
        no_cuda=(not torch.cuda.is_available()),
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # as smaller loss is better
        remove_unused_columns=False,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        save_strategy="epoch",
        save_total_limit=10,  # deletes older checkpoints on reaching this limit
    )

    num_train_steps = int(
        np.ceil(len(dataset["train"]) * num_train_epochs / batch_size)
    )

    optimizer_class = optim.__dict__[optimizer_name]
    if not encoder_optimizer_kwargs:
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    else:
        encoder_params = (
            p[1] for p in model.named_parameters() if p[0].split(".", 1)[0] == "encoder"
        )
        other_params = (
            p[1] for p in model.named_parameters() if p[0].split(".", 1)[0] != "encoder"
        )

        optimizer = optimizer_class(
            [
                {"params": encoder_params, **encoder_optimizer_kwargs},
                {"params": other_params},
            ],
            **optimizer_kwargs,
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
    )

    binary_class = num_labels == 2
    labels = np.array(dataset["train"][label_col])

    if binary_class:
        # pos_weight of torch's BCEWithLogitsLoss expects
        # it in this way i.e. count(neg_class) / count(pos_class)
        # see the example explanation at:
        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        class_counts = np.bincount(labels)
        class_weight = class_counts[0] / class_counts[1]

        if pos_label is None:
            pos_label = 1

        elif isinstance(pos_label, str):
            print(f"Specified pos_label is {pos_label}", end=" ")
            pos_label = dataset["train"].features[label_col].names.index(pos_label)
            print(
                f"which is numerically represented as {pos_label}. "
                f"Setting pos_label to {pos_label} for evaluation."
            )

        prfbeta_kwargs = dict(average="binary", pos_label=pos_label)

    else:
        class_weight = compute_class_weight(
            "balanced", classes=np.unique(labels), y=labels
        )
        prfbeta_kwargs = dict(average="macro")

    compute_metrics = partial(
        compute_aprfbeta,
        prfbeta_kwargs=prfbeta_kwargs,
    )

    trainer = HFTrainer(
        model=model,
        args=train_args,
        optimizers=(optimizer, scheduler),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    if no_class_weight:
        trainer.register_loss_fn(binary_class, weight=None)
    else:
        trainer.register_loss_fn(binary_class, weight=class_weight)

    if num_train_epochs > 0:
        trainer.train()

    dataset_for_metrics = ["train", "validation"]
    if not (-1 in dataset["test"][label_col]):
        dataset_for_metrics.append("test")
    else:
        # nothing to compare, write predictions to a file
        activations = trainer.predict(
            dataset["test"].remove_columns(label_col)
        ).predictions

        pred_labels = to_labels(activations)
        np.savetxt(
            output_dir / "predictions.csv",
            pred_labels,
            fmt="%d",
            header="predictions",
            comments="",
        )

    pred_metrics = {
        i: trainer.predict(dataset[i], metric_key_prefix=i).metrics
        for i in dataset_for_metrics
    }

    wandb.log(pred_metrics)
    print(pred_metrics)


if __name__ == "__main__":
    args = parser.parse_args()

    encoder_training = (
        "untuned_encoder" if args.non_trainable_encoder else "tuned_encoder"
    )

    wandb.login()  # prompts for logging in if not done already
    wandb.init(
        project="ContextGNNBERT",
        name=f"{args.dataset_name}-{args.classifier_net}-{args.encoder_model}-{encoder_training}",
        tags=[
            "ContextGNNBERT",
            args.dataset_name,
            args.classifier_net,
            args.encoder_model,
            encoder_training,
        ],
        group="ContextGNNBERT",
        entity="pensieves",
        config=args,
    )

    dataset_name = args.dataset_name.split(":")
    config_name = dataset_name[1] if len(dataset_name) == 2 else None
    dataset_name = dataset_name[0]

    if len(args.hidden_channels) == 1:
        args.hidden_channels = args.hidden_channels[0]
    if len(args.input_text_cols) == 1:
        args.input_text_cols = args.input_text_cols[0]
    if len(args.context_corpus_splits) == 1:
        args.context_corpus_splits = args.context_corpus_splits[0]

    main(
        dataset_name=dataset_name,
        config_name=config_name,
        label_col=args.label_col,
        pos_label=args.pos_label,
        input_text_cols=args.input_text_cols,
        context_corpus_splits=args.context_corpus_splits,
        context_masker=args.context_masker,
        context_masker_init_kwargs=args.context_masker_init_kwargs,
        context_mask_fn_kwargs=args.context_mask_fn_kwargs,
        truncation_strategy=args.truncation_strategy,
        encoder_model=args.encoder_model,
        trainable_encoder=(not args.non_trainable_encoder),
        classifier_net=args.classifier_net,
        gnn_kwargs=args.gnn_kwargs,
        dropout=args.dropout,
        num_layers=args.num_layers,
        hidden_channels=args.hidden_channels,
        non_linearity=args.non_linearity,
        load_checkpoint=args.load_checkpoint,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        optimizer_name=args.optimizer_name,
        optimizer_kwargs=args.optimizer_kwargs,
        encoder_optimizer_kwargs=args.encoder_optimizer_kwargs,
        num_warmup_steps=args.num_warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        no_class_weight=args.no_class_weight,
    )
