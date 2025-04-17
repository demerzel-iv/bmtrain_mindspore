from mindspore.experimental import optim

class WarmupStableDecayLRScheduler(optim.lr_scheduler.LRScheduler):
    """
    A learning rate scheduler that implements a warmup phase, a stable phase, 
    and a decay phase. The learning rate increases linearly during the warmup 
    phase, remains constant during the stable phase, and decreases linearly 
    during the decay phase.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        last_epoch (int): The index of the last epoch. Default: -1.
        warmup_steps (int): The number of steps for the warmup phase. Default: 100.
        decay_start (int): The step at which the decay phase starts. Default: 4000.
        total_iters (int): The total number of iterations. Default: 5000.
        lr (float): The initial learning rate. Default: 1e-5.
    """
    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        warmup_steps=100,
        decay_start=4000,
        total_iters=5000,
        lr=1e-5
    ):
        self.warmup_steps = warmup_steps
        self.decay_start = decay_start
        self.total_iters = total_iters
        self.lr = lr
        super(WarmupStableDecayLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rate for the current step.

        Returns:
            list[float]: A list containing the learning rate for each parameter group.
        """
        if self.last_epoch < self.warmup_steps:
            # Warmup
            lr_ret = self.lr * (self.last_epoch + 1) / self.warmup_steps
        elif self.last_epoch < self.decay_start:
            # Stable
            lr_ret = self.lr
        else:
            # Decay
            decay_iters = self.total_iters - self.decay_start
            lr_ret = self.lr * max(0, 1 - (self.last_epoch - self.decay_start) / decay_iters)

        return [lr_ret] * len(self._last_lr)
