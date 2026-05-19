import zlib
from typing import Any

import jax
import numpy as np
from jax._src import array, core, sharding_impls
from jax._src.interpreters import pxla
from jax.experimental.multihost_utils import _psum, host_local_array_to_global_array
from jax.sharding import PartitionSpec as P

"""
These functions are copied from jax.experimental.multihost_utils but for syncing 
we want to give it a mesh instead of global since we have some processes which are train workers
and some are inference workers. So we want the option to only sync workers on the same mesh
"""


def sync_over_mesh(name: str, mesh: jax.sharding.Mesh | None = None):
    h = np.uint32(zlib.crc32(name.encode()))
    assert_equal(h, mesh, f"sync_global_devices name mismatch ('{name}')")


def assert_equal(in_tree, mesh, fail_message: str = ""):
    """verifies that all the hosts have the same tree of values."""

    def concat_in_tree(x):
        if isinstance(x, array.ArrayImpl) and not x.is_fully_addressable:
            return np.asarray(x.addressable_data(0))
        else:
            x = np.asarray(x)
            if x.ndim == 0:
                x = np.expand_dims(x, axis=0)
            n_processes = jax.process_count() if mesh is None else mesh.axis_sizes[0]
            return np.concat([x] * n_processes)

    out = process_allgather_over_mesh(in_tree, True, mesh)
    expected_in_tree = jax.tree.map(concat_in_tree, in_tree)
    if not jax.tree.all(jax.tree.map(lambda *x: np.all(np.equal(*x)), expected_in_tree, out)):
        raise AssertionError(f"{fail_message}. expected: {out}; got: {in_tree}.")


def process_allgather_over_mesh(in_tree: Any, tiled, mesh: jax.sharding.Mesh | None = None) -> Any:
    def _pjit(inp):
        return _handle_array_process_allgather(inp, tiled, mesh)

    return jax.tree.map(_pjit, in_tree)


def _handle_array_process_allgather(inp, tiled, global_mesh: jax.sharding.Mesh | None = None):
    if isinstance(inp, array.ArrayImpl) and not inp.is_fully_addressable:
        if not tiled:
            raise ValueError("Gathering global non-fully-addressable arrays only supports tiled=True")
        if isinstance(inp.sharding, sharding_impls.NamedSharding):
            reps = inp.sharding.update(spec=P())
        else:
            reps = sharding_impls.GSPMDSharding.get_replicated(
                inp.sharding._device_assignment, memory_kind=inp.sharding.memory_kind
            )
        out = jax.jit(lambda x: x, out_shardings=reps)(inp)
    else:
        # All inputs here will be fully addressable.
        if jax.process_count() == 1:
            out = np.asarray(inp)
            return np.expand_dims(out, axis=0) if not tiled else out

        if global_mesh is None:
            devices = np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count())
            global_mesh = jax.sharding.Mesh(devices, ("processes", "local_devices"))
        else:
            assert global_mesh.axis_names == ("processes", "local_devices"), (
                f"Expected global_mesh axis names to be ('processes', 'local_devices'), got {global_mesh.axis_names}"
            )

        pspec = P("processes")
        s = jax.sharding.NamedSharding(global_mesh, pspec)

        host_np_arr = np.asarray(inp)
        if host_np_arr.ndim == 0 or not tiled:
            host_np_arr = np.expand_dims(host_np_arr, axis=0)

        aval = core.ShapedArray(host_np_arr.shape, host_np_arr.dtype)
        global_aval = pxla.mesh_local_to_global(global_mesh, pxla.get_array_mapping(pspec), aval)  # type: ignore

        bufs = [jax.device_put(host_np_arr, d) for d in jax.local_devices()]
        global_arr = array.make_array_from_single_device_arrays(global_aval.shape, s, bufs)
        with jax.set_mesh(global_mesh):
            out = jax.jit(lambda x: x, out_shardings=P())(global_arr)
    return np.asarray(out.addressable_data(0))


def broadcast_over_mesh(in_tree: Any, is_source: bool | None = None, mesh: jax.sharding.Mesh | None = None) -> Any:
    """Broadcast data from a source host (host 0 by default) to all other hosts.

    Args:
      in_tree: pytree of arrays - each array *must* have the same shape across the
        hosts.
      is_source: optional bool denoting whether the caller is the source. Only
        'source host' will contribute the data for the broadcast. If None, then
        host 0 is used.

    Returns:
      A pytree matching in_tree where the leaves now all contain the data from the
      first host.
    """
    if jax.process_count() == 1:
        return jax.tree.map(np.asarray, in_tree)

    if is_source is None:
        is_source = jax.process_index() == 0

    devices: np.ndarray = np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count())

    if mesh is None:
        global_mesh = jax.sharding.Mesh(devices, ("processes", "local_devices"))
    else:
        global_mesh = mesh

    pspec = P("processes")

    def pre_jit(x):
        if is_source:
            inp = x
        else:
            inp = np.zeros_like(x)
        inp = np.expand_dims(inp, axis=0)
        return host_local_array_to_global_array(inp, global_mesh, pspec)

    def post_jit(x):
        return jax.device_get(x.addressable_data(0))

    in_tree = jax.tree.map(pre_jit, in_tree)
    with jax.set_mesh(global_mesh):
        out_tree = jax.jit(_psum, out_shardings=P())(in_tree)

    return jax.tree.map(post_jit, out_tree)
