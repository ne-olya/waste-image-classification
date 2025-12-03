import torch


class ClassificationMeter:
    def __init__(self, num_classes):
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, logits, target):
        prediction = logits.argmax(1).cpu()
        target = target.cpu()
        n = len(self.matrix)
        self.matrix += torch.bincount(n * target + prediction, minlength=n * n).reshape(n, n)

    def compute(self):
        matrix = self.matrix.float()
        tp = matrix.diag()
        precision = tp / matrix.sum(0).clamp_min(1)
        recall = tp / matrix.sum(1).clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        return {
            "accuracy": float(tp.sum() / matrix.sum().clamp_min(1)),
            "macro_f1": float(f1.mean()),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "confusion_matrix": self.matrix.tolist(),
        }


def expected_calibration_error(logits, target, bins=15):
    probabilities = logits.softmax(1)
    confidence, prediction = probabilities.max(1)
    correct = prediction.eq(target)
    ece = torch.zeros((), device=logits.device)
    boundaries = torch.linspace(0, 1, bins + 1, device=logits.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += (
                selected.float().mean()
                * (confidence[selected].mean() - correct[selected].float().mean()).abs()
            )
    return float(ece)
