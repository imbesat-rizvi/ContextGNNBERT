import json
import torch
from pathlib import Path


def mask_averaging(seq, mask):
    r"""
    averages vectors in seq according to mask which may have one less dimension
    corresponding to the vector length i.e. seq is BxTxD and mask can either be
    BxTxD or BxT
    """

    if mask.shape != seq.shape:
        # expand mask dimension and repeat mask value
        # i.e. convert mask from BxT to BxTxD
        mask = mask.unsqueeze(-1).expand(-1, -1, seq.shape[-1])

    masked_seq_sum = (mask * seq).sum(dim=1)  # masked sum across T
    masked_avg = masked_seq_sum / mask.sum(dim=1)

    return masked_avg


def update_model_state_dict(model, saved_path="", load_strategy="best"):

    load_strategies = ("best", "latest")
    assert (
        load_strategy in load_strategies
    ), f"load_strategy should be one among {load_strategies}"

    if saved_path:
        model_path = Path(saved_path)
        if model_path.is_dir():
            if (model_path / "pytorch_model.bin").exists():

                print(
                    "Given path is a checkpoint directory. Loading from pytorch_model.bin"
                )

                model_path = model_path / "pytorch_model.bin"

            else:
                print(
                    f"Given path is a directory with multiple checkpoints. Loading from `load_strategy={load_strategy}` checkpoint."
                )

                checkpoints = [
                    int(f.name.split("-")[-1]) for f in model_path.glob("checkpoint*")
                ]

                latest_checkpoint = max(checkpoints)
                model_dir = model_path / f"checkpoint-{latest_checkpoint}"
                model_path = model_dir / "pytorch_model.bin"

                if load_strategy == "best":
                    with open(model_dir / "trainer_state.json", "r") as f:
                        best_checkpoint_dir = json.load(f)["best_model_checkpoint"]
                        model_path = Path(best_checkpoint_dir) / "pytorch_model.bin"

        state_dict = torch.load(model_path, map_location="cpu")
        load_message = model.load_state_dict(state_dict, strict=False)
        print(f"Strict load was set as `false`. Load message is {load_message}")

    return model
