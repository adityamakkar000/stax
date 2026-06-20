import asyncio
import os
import pickle
import shutil
import tempfile
import threading
import time
from typing import Any

import jax
import tensorflow as tf
from etils import epath

tf.config.set_visible_devices([], 'TPU')
tf.config.set_visible_devices([], 'GPU')

from stax.logger import staxLogger as logger
from stax.multihost_utils import process_allgather_over_mesh, sync_over_mesh
from stax.utils import get_rank

GB = 1024**3
CONCURRENT_CHUNK_LIMIT = 8

def parent_dir(filename):
    return filename.rsplit('/', 1)[0]

def name(filename):
    return filename.rsplit('/', 1)[1]

class Checkpoint:
    def __init__(self, filename):
        self._filename = filename
        self._values = {}

    def __setattr__(self, name, value):
        if name in ('exists', 'save', 'load'):
            return super().__setattr__(name, value)
        if name.startswith('_'):
            return super().__setattr__(name, value)
        self._values[name] = value

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._values[name]
        except KeyError:
            raise ValueError(name)

    def save(self, filename=None, keys=None):
        assert self._filename or filename
        filename = filename or self._filename
        logger.info(f'[checkpointer] Writing chunked checkpoint to directory: {filename}')
        self._save(filename, keys)

    def _save(self, filename, keys):
        start_time = time.time()
        keys = tuple(self._values.keys() if keys is None else keys)
        assert all([not k.startswith('_') for k in keys]), keys
        data = {k: self._values[k] for k in keys}
        data['_timestamp'] = time.time()

        flat_data, treedef = jax.tree_util.tree_flatten(data)

        check_dir = "/tmp" if 'gs://' in filename else filename
        if 'gs://' not in filename:
            os.makedirs(filename, exist_ok=True)

        free_space = shutil.disk_usage(check_dir).free
        max_concurrent_chunks = max(1, min(CONCURRENT_CHUNK_LIMIT, int(free_space * 0.25 / GB)))
        logger.info(f"[checkpointer] Allowed storage: {free_space * 0.25 / GB:.2f}GB. Limiting concurrency to {max_concurrent_chunks} chunks.")

        if 'gs://' in filename:
            tf.io.gfile.makedirs(filename)

        async def _upload_all_chunks():
            sem = asyncio.Semaphore(max_concurrent_chunks)

            async def sem_worker(func, *args):
                async with sem:
                    return await asyncio.to_thread(func, *args)

            chunk_idx = 0
            current_chunk = []
            current_size = 0
            MAX_CHUNK_BYTES = GB
            tasks = []

            for item in flat_data:
                item_size = getattr(item, 'nbytes', 8)

                if current_size + item_size > MAX_CHUNK_BYTES and current_chunk:
                    tasks.append(lambda idx, total_chunks: sem_worker(
                        self._write_and_upload_chunk, filename, idx, current_chunk, total_chunks
                    ))
                    chunk_idx += 1
                    current_chunk = []
                    current_size = 0

                current_chunk.append(item)
                current_size += item_size

            if current_chunk:
                tasks.append(lambda idx, total_chunks: sem_worker(
                    self._write_and_upload_chunk, filename, idx, current_chunk, total_chunks
                ))

            for i in range((total_chunks := len(tasks))):
                tasks[i] = tasks[i](i, total_chunks)

            def _write_treedef():
                treedef_path = f"{filename}/treedef.pkl"
                if 'gs://' in filename:
                    with tf.io.gfile.GFile(treedef_path, 'wb') as f:
                        pickle.dump(treedef, f)
                else:
                    with open(treedef_path, 'wb') as f:
                        pickle.dump(treedef, f)

            tasks.append(asyncio.to_thread(_write_treedef))

            await asyncio.gather(*tasks)

        asyncio.run(_upload_all_chunks())

        elapsed = time.time() - start_time
        logger.info(f'[checkpointer] Successfully wrote chunked checkpoint asynchronously in {elapsed:.3f}s.')

    def _write_and_upload_chunk(self, base_dir, chunk_idx, chunk_data, total_chunks):
        logger.info(f"[checkpointer] Writing chunk {chunk_idx + 1}/{total_chunks}...")
        fd, tmp_local = tempfile.mkstemp(prefix=f"chunk_{chunk_idx}_", suffix=".pkl", dir="/tmp")
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            dest_path = f"{base_dir}/chunk_{chunk_idx}.pkl"
            if 'gs://' in base_dir:
                tf.io.gfile.copy(tmp_local, dest_path, overwrite=True)
            else:
                shutil.move(tmp_local, dest_path)
        finally:
            os.remove(tmp_local)

    def load_as_dict(self, filename=None, max_concurrent_chunks: int = 4):
        assert self._filename or filename
        filename = filename or self._filename
        logger.info(f'[checkpointer] Reading chunked checkpoint from directory: {filename}')
        return asyncio.run(self._load_as_dict_async(filename, max_concurrent_chunks))

    async def _load_as_dict_async(self, filename, max_concurrent_chunks):
        is_gcs = 'gs://' in filename

        def _read_bytes(path):
            if is_gcs:
                with tf.io.gfile.GFile(path, 'rb') as f:
                    return f.read()
            else:
                with open(path, 'rb') as f:
                    return f.read()

        # first load the tree metadata blocking
        treedef_path = f"{filename}/treedef.pkl"
        treedef = pickle.loads(await asyncio.to_thread(_read_bytes, treedef_path))

        
        def _list_chunk_indices():
            entries = tf.io.gfile.listdir(filename) if is_gcs else os.listdir(filename)
            indices = []
            for entry in entries:
                stripped = entry.rstrip('/')
                if stripped.startswith('chunk_') and stripped.endswith('.pkl'):
                    indices.append(int(stripped[len('chunk_'):-len('.pkl')]))
            return sorted(indices)

        chunk_indices = await asyncio.to_thread(_list_chunk_indices)
        if chunk_indices != list(range(len(chunk_indices))):
            raise ValueError(
                f"Checkpoint at {filename} has missing/unexpected chunks; "
                f"found indices {chunk_indices}, expected a contiguous range "
                f"starting at 0."
            )

        sem = asyncio.Semaphore(max(1, max_concurrent_chunks))

        async def _load_chunk(chunk_idx):
            chunk_path = f"{filename}/chunk_{chunk_idx}.pkl"
            async with sem:
                logger.info(f"[checkpointer] Loading chunk {chunk_idx + 1}/{len(chunk_indices)}...")
                raw = await asyncio.to_thread(_read_bytes, chunk_path)
                return pickle.loads(raw)

        start = time.time() 
        chunks = await asyncio.gather(*[_load_chunk(idx) for idx in chunk_indices])
        end = time.time()

        flat_data = []
        for chunk in chunks:
            flat_data.extend(chunk)

        data = jax.tree_util.tree_unflatten(treedef, flat_data)

        step = name(filename)
        logger.info(f'[checkpointer] Loaded chunked checkpoint from {step} in {end - start:.3f}s.')
        return data

class OldCheckpointer:
    def __init__(
        self,
        output_dir: str,
        max_to_keep: int = 1,
        *,
        train_mesh: jax.sharding.Mesh | None = None,
    ):
        if not output_dir.startswith("gs"):
            raise AssertionError("output_dir must be a valid GCS path starting with gs")

        self.output_dir = output_dir
        if not self.output_dir.endswith('/'):
            self.output_dir = self.output_dir + '/'
        self.max_to_keep = max_to_keep
        self.train_mesh = train_mesh
        self.directory = epath.Path(self.output_dir)
        self.rank = get_rank()
        self.checkpoint_thread: threading.Thread | None = None

        if self.rank == 0 and not self.directory.exists():
            logger.info(f"[checkpointer] Creating checkpoint directory at {self.directory}")
            self.directory.mkdir(parents=True, exist_ok=True)

    def block_until_ready(self):
        if self.checkpoint_thread is not None:
            self.checkpoint_thread.join()
            self.checkpoint_thread = None

    def save(self, step: int, checkpoint_data: dict, metadata: dict | None = None):
        def _maybe_gather(x):
            if isinstance(x, jax.Array):
                if x.is_fully_addressable:
                    return jax.device_get(x)
                return process_allgather_over_mesh(x, tiled=True, mesh=self.train_mesh)
            return x

        checkpoint_data = jax.tree.map(_maybe_gather, checkpoint_data)
        data = {
            "checkpoint_data": checkpoint_data,
            "metadata": metadata or {},
        }

        self.block_until_ready()

        if self.rank == 0:
            filename = f"{self.output_dir}{step}"
            def _write():
                checkpoint = Checkpoint(filename)
                checkpoint.data = data
                checkpoint.save()
                del checkpoint
                self._maybe_delete_old_checkpoints()

            self.checkpoint_thread = threading.Thread(target=_write, daemon=True)
            self.checkpoint_thread.start()

        sync_over_mesh("checkpoint_save_sync", self.train_mesh)

    def restore(self, *, step: int | None = None):
        if step is None:
            step = self.latest_step
            if step is None:
                logger.info("[checkpointer] No checkpoint found to restore.")
                return None

        path_name = f"{self.output_dir}{step}"
        checkpoint = Checkpoint(path_name)
        data : dict[str, Any] = checkpoint.load_as_dict()['data']
        del checkpoint
        return data["checkpoint_data"], data["metadata"]

    def _maybe_delete_old_checkpoints(self):
        if self.rank == 0:
            dirs = [f for f in self.directory.iterdir() if f.is_dir() and f.name.isdigit()]
            dirs = sorted(dirs, key=lambda x: int(x.name))
            if self.max_to_keep > 0 and len(dirs) > self.max_to_keep:
                to_delete = dirs[:-self.max_to_keep]
                for d in to_delete:
                    logger.info(f"[checkpointer] Deleting old checkpoint directory: {d}")
                    d.rmtree()

    @property
    def latest_step(self) -> int | None:
        """Get the latest checkpoint."""
        if not self.directory.exists():
            return None
        dirs = [f for f in self.directory.iterdir() if f.is_dir() and f.name.isdigit()]
        dirs = sorted(dirs, key=lambda x: int(x.name))
        if not dirs:
            return None
        return int(dirs[-1].name)