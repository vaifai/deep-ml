# --- cell 1 ---
import torch

class MyTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, 28, 28) float tensor in [0,1]
        Return: transformed tensor, same shape/dtype.
        Must be non-identity and deterministic.
        """
        # TODO: implement your custom transformation logic here
        eps = 1e-6
        mean = x.mean()
        std = x.std(unbiased=False).clamp(min=eps)
        x_norm = (x - mean)/std
        return x_norm