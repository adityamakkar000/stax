import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Array


pytree = {
    'a': jnp.array([1, 2, 3]),
    'b': {
        'c': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'd': jnp.array(5)
    },
    'e': (jnp.array([10, 20]), jnp.array([[30, 40], [50, 60]]))
}

sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

breakpoint()
print(PyTree(sharding))
jax.tree.map(
    lambda x,p: print(x, " " , p), 
    pytree,
    sharding
)