import os
import pickle
import shutil
import time
from typing import Any

import jax
from etils import epath

from stax.logger import staxLogger as logger
from stax.multihost_utils import process_allgather_over_mesh, sync_over_mesh
from stax.utils import get_rank

############
# A checkpointer class for flax models, compatible with saving/loading from gs:// buckets.
# Taken from: https://github.com/danijar/elements/blob/main/elements/checkpoint.py
############

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
        content = pickle.dumps(data)
        if 'gs://' in filename:
            import tensorflow as tf
            tf.io.gfile.makedirs(parent_dir(filename))
            with tf.io.gfile.GFile(filename, 'wb') as f:
                f.write(content)
        else:
            os.makedirs(filename, exist_ok=True)
            tmp = parent_dir(filename) + '/' + name(filename) + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(content)
            shutil.move(tmp, filename)
        elapsed = time.time() - start_time
        logger.info(f'Wrote checkpoint in {elapsed:.3f}s.')

    def load_as_dict(self, filename=None):
        assert self._filename or filename
        filename = filename or self._filename
        if 'gs://' in filename:
            import tensorflow as tf
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
        train_mesh : jax.sharding.Mesh | None = None,
    ): 

        """Initialize the checkpointer.

        Args:
            output_dir (str): Directory where checkpoints will be saved. Must be a valid GCS path starting with gs://.
        """
        if not output_dir.startswith("gs"):
            raise AssertionError("output_dir must be a valid GCS path starting with gs")

        self.output_dir = output_dir
        if not self.output_dir.endswith('/'):
            self.output_dir = self.output_dir + '/'
        self.max_to_keep = max_to_keep
        self.train_mesh = train_mesh
        self.directory = epath.Path(self.output_dir)
        self.rank = get_rank()

        if self.rank == 0 and not self.directory.exists():
            logger.info(f"Creating checkpoint directory at {self.directory}")
            self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, step: int, checkpoint_data: dict, metadata: dict | None = None):        
        """Save a checkpoint."""

        checkpoint_data = process_allgather_over_mesh(
            checkpoint_data, 
            tiled=True, 
            mesh=self.train_mesh
        )

        data = {
            "checkpoint_data": checkpoint_data,
            "metadata": metadata or {},
        }

        if self.rank == 0:
            filename = f"{self.output_dir}{step}"
            checkpoint = Checkpoint(filename)
            checkpoint.data = data
            checkpoint.save()
            del checkpoint

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
