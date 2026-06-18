import os
import pickle
import shutil
import time
from typing import Any

import jax
import tensorflow as tf

tf.config.set_visible_devices([], 'TPU')
tf.config.set_visible_devices([], 'GPU')

from stax.logger import staxLogger as logger


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
        logger.info(f'Writing chunked checkpoint to directory: {filename}')
        self._save(filename, keys)

    def _save(self, filename, keys):
        start_time = time.time()
        keys = tuple(self._values.keys() if keys is None else keys)
        assert all([not k.startswith('_') for k in keys]), keys
        
        data = self._values
        data['_timestamp'] = time.time()
        
        flat_data, treedef = jax.tree_util.tree_flatten(data)
        
        chunk_idx = 0
        current_chunk = []
        current_size = 0
        # store 5GB chunks
        MAX_CHUNK_BYTES = 5 * 1024**3
        
        if 'gs://' in filename:
            tf.io.gfile.makedirs(filename)
        else:
            os.makedirs(filename, exist_ok=True)
            
        for item in flat_data:
            item_size = getattr(item, 'nbytes', 8) 
            
            if current_size + item_size > MAX_CHUNK_BYTES and current_chunk:
                self._write_and_upload_chunk(filename, chunk_idx, current_chunk)
                chunk_idx += 1
                current_chunk = []
                current_size = 0
                
            current_chunk.append(item)
            current_size += item_size
            
        if current_chunk:
            self._write_and_upload_chunk(filename, chunk_idx, current_chunk)
            
        treedef_path = f"{filename}/treedef.pkl"
        if 'gs://' in filename:
            with tf.io.gfile.GFile(treedef_path, 'wb') as f:
                pickle.dump(treedef, f)
        else:
            with open(treedef_path, 'wb') as f:
                pickle.dump(treedef, f)

        elapsed = time.time() - start_time
        logger.info(f'Successfully wrote chunked checkpoint in {elapsed:.3f}s.')

    def _write_and_upload_chunk(self, base_dir, chunk_idx, chunk_data):
        logger.info(f"Processing chunk {chunk_idx}...")
        tmp_local = f"/tmp/chunk_{chunk_idx}.pkl"
        
        with open(tmp_local, 'wb') as f:
            pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        dest_path = f"{base_dir}/chunk_{chunk_idx}.pkl"
        
        if 'gs://' in base_dir:
            tf.io.gfile.copy(tmp_local, dest_path, overwrite=True)
            os.remove(tmp_local) 
        else:
            shutil.move(tmp_local, dest_path)

    def load_as_dict(self, filename=None):
        assert self._filename or filename
        filename = filename or self._filename
        
        treedef_path = f"{filename}/treedef.pkl"
        if 'gs://' in filename:
            with tf.io.gfile.GFile(treedef_path, 'rb') as f:
                treedef = pickle.loads(f.read())
        else:
            with open(treedef_path, 'rb') as f:
                treedef = pickle.loads(f.read())
                
        flat_data = []
        chunk_idx = 0
        while True:
            chunk_path = f"{filename}/chunk_{chunk_idx}.pkl"
            
            if 'gs://' in filename:
                if not tf.io.gfile.exists(chunk_path):
                    break
                logger.info(f"Loading {chunk_path}...")
                with tf.io.gfile.GFile(chunk_path, 'rb') as f:
                    flat_data.extend(pickle.loads(f.read()))
            else:
                if not os.path.exists(chunk_path):
                    break
                logger.info(f"Loading {chunk_path}...")
                with open(chunk_path, 'rb') as f:
                    flat_data.extend(pickle.loads(f.read()))
                    
            chunk_idx += 1
            
        data = jax.tree_util.tree_unflatten(treedef, flat_data)
        
        age = time.time() - data.get('_timestamp', time.time())
        logger.info(f'Loaded chunked checkpoint from {age:.0f} seconds ago.')
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
