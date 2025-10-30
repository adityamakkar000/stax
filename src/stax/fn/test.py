import jax
from flax import linen as nn
from jaxtyping import Array
import jax.numpy as jnp
import optax
from stax.fn.main import get_steps_fn

if __name__ == "__main__":

    class testNN(nn.Module):
        @nn.compact
        def __call__(self, x: Array) -> Array:
            return nn.Dense(1)(x)

    def loss_fn(x: Array) -> Array:
        return (x**2).sum()

    def step(model, params, x):
        return loss_fn(model.apply({"params": params}, x))

    x_init = jnp.zeros((1, 16), dtype=jnp.float32)
    model = testNN()
    params = model.init(rngs=jax.random.key(0), x=x_init)["params"]

    tx = optax.adamw(learning_rate=1e-4)
    opt_state = tx.init(params)

    train_step, val_step = get_steps_fn(step, model, tx, grad_steps=1, has_aux=False)

    x_input = jnp.zeros((1, 4, 16))

    output = train_step(params, opt_state, x_input)
    output_val = val_step(params, x_input)

    breakpoint()
