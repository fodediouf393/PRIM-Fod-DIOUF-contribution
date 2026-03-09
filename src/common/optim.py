import torch
from torch.optim import AdamW

class WarmupToMaxLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_start: float, lr_max: float, warmup_epochs: int, last_epoch: int = -1):
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.warmup_epochs = max(1, warmup_epochs)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch <= self.warmup_epochs:
            t = epoch / self.warmup_epochs
            lr = self.lr_start + t * (self.lr_max - self.lr_start)
        else:
            lr = self.lr_max
        return [lr for _ in self.base_lrs]

def build_optimizer_and_scheduler(model, lr_start=1e-4, lr_max=1e-2, warmup_epochs=10, weight_decay=1e-4):
    opt = AdamW(model.parameters(), lr=lr_start, weight_decay=weight_decay)
    sch = WarmupToMaxLR(opt, lr_start=lr_start, lr_max=lr_max, warmup_epochs=warmup_epochs)
    return opt, sch
