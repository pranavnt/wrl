from typing import Any, Callable, Dict, Sequence, Union

import flax
import jax.numpy as jnp
import numpy as np

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]
Array = Union[np.ndarray, jnp.ndarray]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
ModuleMethod = Union[str, Callable, None]
