from typing import Any, Dict
import torchvision.transforms.v2 as T


class NaNtoNum(T.Transform):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params

    def _transform(self, x: Any) -> Any:
        return x.nan_to_num(**self.params)
