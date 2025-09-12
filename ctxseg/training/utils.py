import optax
from flax import nnx

class EMAOptimizer(nnx.Optimizer):    
    def __init__(
        self,
        model: nnx.Module,
        tx: optax.GradientTransformation,
        wrt: nnx.filterlib.Filter = nnx.Param,
        ema_decay: float = 0.999,
    ):
        super().__init__(model, tx, wrt)
        self.ema_model = model
        self.ema_decay = ema_decay
        self.wrt = wrt

    def update(self, grads, **kwargs):
        super().update(grads, **kwargs)
        ema_params = nnx.split(self.ema_model, self.wrt, ...)[1]
        graphdef, new_params, *other_state = nnx.split(self.model, self.wrt, ...)
        ema_params = optax.incremental_update(ema_params, new_params, self.ema_decay)
        self.ema_model = nnx.merge(graphdef, ema_params, *other_state)


class LossLogs:
    def __init__(self):
        self.loss_logs = {}

    def update(self, losses):
        from flax import nnx
        for k, v in losses.items():
            if not k in self.loss_logs:
                self.loss_logs[k] = nnx.metrics.Average()
            self.loss_logs[k].update(values=v)

    def __repr__(self):
        return f",".join([f"{k}:{v.compute():.4f}" for k,v in self.loss_logs.items()])

    def compute(self):
        return {k: v.compute() for k,v in self.loss_logs.items()}

