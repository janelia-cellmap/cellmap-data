from typing import Any, Dict
import torchvision.transforms.v2 as T
import torch


class Binarize(T.Transform):
    """Binarize input tensors using a threshold-based transformation.

    This transformation converts input tensors to binary values by applying
    a threshold comparison. Values above the threshold become 1 (or dtype max),
    values below become 0. NaN values are preserved in the output when present
    in floating-point tensors.

    Args:
        threshold: Threshold value for binarization. Defaults to 0.
            Values greater than threshold become 1, others become 0.
    """

    def __init__(self, threshold=0) -> None:
        """Initialize the binarization transformation.

        Args:
            threshold: Threshold value for the binary comparison. Defaults to 0.
                Values strictly greater than this threshold become 1, others become 0.
        """
        super().__init__()
        self.threshold = threshold

    def _transform(self, inpt: Any, params: Dict[str, Any] | None = None) -> Any:
        """Apply binarization transformation to input tensor.

        Args:
            inpt: Input tensor to binarize. Should be numeric tensor.
            params: Additional parameters (not used, maintained for interface compatibility).

        Returns:
            Binarized tensor where values > threshold become 1 and others become 0.
            NaN values are preserved when present in floating-point inputs.
        """
        out = (inpt > self.threshold).to(inpt.dtype)
        hasnans = inpt.isnan().any()
        if not torch.is_floating_point(out) and hasnans:
            out = out.to(torch.float32)
        if hasnans:
            out[inpt.isnan()] = torch.nan
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"

    def transform(self, x: Any, params: Dict[str, Any] | None = None) -> Any:
        return self._transform(x, params)
