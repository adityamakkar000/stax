import os
import pickle
import shutil
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
        except AttributeError:
            raise ValueError(name)
        
    def save(self, filename=None, keys=None):
        assert self._filename or filename
        filename = filename or self._filename
        logger.info(f'Writing checkpoint: {filename}')
        self._save(filename, keys)

    def _save(self, filename, keys):
        start_time = time.time()
        keys = tuple(self._values.keys() if keys is None else keys)
        assert all([not k.startswith('_') for k in keys]), keys
        data = self._values
        data['_timestamp'] = time.time()
        
        if 'gs://' in filename:
            tf.io.gfile.makedirs(parent_dir(filename))
            tmp_local = '/tmp/' + name(filename) + '.tmp'
            with open(tmp_local, 'wb') as f:
                pickle.dump(data, f)
            
            tf.io.gfile.copy(tmp_local, filename, overwrite=True)
            os.remove(tmp_local)
        else:
            os.makedirs(filename, exist_ok=True)
            tmp = parent_dir(filename) + '/' + name(filename) + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump(data, f)
            shutil.move(tmp, filename)
            
        elapsed = time.time() - start_time
        logger.info(f'Wrote checkpoint in {elapsed:.3f}s.')

    def load_as_dict(self, filename=None):
        assert self._filename or filename
        filename = filename or self._filename
        if 'gs://' in filename:
            with tf.io.gfile.GFile(filename, 'rb') as f:
                data = pickle.loads(f.read())
        else:
            with open(filename, 'rb') as f:
                data = pickle.loads(f.read())
        age = time.time() - data['_timestamp']
        logger.info(f'Loaded checkpoint from {age:.0f} seconds ago.')
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
            logger.info(f"Creating checkpoint directory at {self.directory}")
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
                logger.info("No checkpoint found to restore.")
                return None

        path_name = f"{self.output_dir}{step}"
        checkpoint = Checkpoint(path_name)
        data : dict[str, Any] = checkpoint.load_as_dict()['data']
        del checkpoint
        return data["checkpoint_data"], data["metadata"]

    def _maybe_delete_old_checkpoints(self):
        if self.rank == 0:
            files = [f for f in self.directory.iterdir() if f.is_file()]
            files = sorted(files, key=lambda x: int(x.name))
            if self.max_to_keep > 0 and len(files) > self.max_to_keep:
                to_delete = files[:-self.max_to_keep]
                for f in to_delete:
                    logger.info(f"Deleting old checkpoint: {f}")
                    f.unlink()

    @property
    def latest_step(self) -> int | None:
        """Get the latest checkpoint."""
        if not self.directory.exists():
            return None
        files = [f for f in self.directory.iterdir() if f.is_file()]
        files = sorted(files, key=lambda x: int(x.name))
        if not files:
            return None
        return int(files[-1].name)
