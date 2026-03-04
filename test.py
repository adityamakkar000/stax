import jax
import jax.numpy as jnp
from flax import linen as nn


class BigModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        DenseStack = nn.remat_scan(nn.Dense, lengths=(10, 10))
        # 100x dense with O(sqrt(N)) memory for gradient computation
        return DenseStack(8, name="dense_stack")(x)


test = BigModel()
params = test.init(jax.random.key(0), jnp.ones((2, 8)))

breakpoint()
