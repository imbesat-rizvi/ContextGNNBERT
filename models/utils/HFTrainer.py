import torch
from transformers import Trainer


class HFTrainer(Trainer):

    def register_loss_fn(self, binary_class=True, weight=None):
        if weight:
            if isinstance(weight, (int, float)):
                weight = [weight]
            
            weight = torch.tensor(
                weight, dtype=torch.float, device=self.model.encoder.device,
            )

        if binary_class:
            loss_fn = torch.nn.BCEWithLogitsLoss
            kwargs = {"pos_weight": weight}
        else:
            loss_fn = torch.nn.CrossEntropyLoss
            kwargs = {"weight": weight}

        self.loss_fn = loss_fn(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get("logits")

        labels = inputs["labels"]
        if logits.shape[1] == 1:
            # BCEWithLogitsLoss wierdly needs labels as floats
            labels = labels * 1.
            logits = logits.view(-1)

        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
