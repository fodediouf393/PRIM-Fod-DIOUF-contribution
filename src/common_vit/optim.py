import torch

def build_optimizer_and_scheduler(model, lr_start, lr_max, warmup_epochs, weight_decay):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_start, weight_decay=weight_decay)

    # linear warmup to lr_max, then cosine decay to lr_start/10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs) * (lr_max / lr_start)
        # cosine
        import math
        t = (epoch - warmup_epochs) / max(1, (100 - warmup_epochs))
        min_lr = lr_start / 10.0
        return (min_lr / lr_start) + 0.5 * (lr_max / lr_start - min_lr / lr_start) * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler
