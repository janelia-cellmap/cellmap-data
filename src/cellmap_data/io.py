import os
import zarr
import tensorstore as ts
import s3fs
from upath import UPath


def open_zarr(path, mode="r"):
    """Open a zarr array from local or S3 path. Handles S3 and local transparently."""
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=True)
        # Remove zarr:// and s3:// prefixes for S3Map
        s3_path = path.replace("zarr://", "").replace("s3://", "")
        store = s3fs.S3Map(root=s3_path, s3=fs, check=False)
        return zarr.open(store, mode=mode)
    else:
        return zarr.open(path, mode=mode)


def open_tensorstore(path, mode="r", concurrency_limit=None):
    """Open a tensorstore array from local or S3 path. Handles concurrency and filetype detection."""
    filetype = "zarr" if ".zarr" in path else "n5"
    if path.startswith("s3://"):
        parts = path.split("/")
        bucket = parts[2]
        s3_path = "/".join(parts[3:])
        kvstore = {
            "driver": "s3",
            "bucket": bucket,
            "path": s3_path,
            "aws_credentials": {"anonymous": True},
        }
    else:
        kvstore = {"driver": "file", "path": os.path.normpath(path)}
    spec = {"driver": filetype, "kvstore": kvstore}
    if concurrency_limit:
        spec["context"] = {
            "data_copy_concurrency": {"limit": concurrency_limit},
            "file_io_concurrency": {"limit": concurrency_limit},
        }
    try:
        if mode == "r":
            dataset_future = ts.open(spec, read=True, write=False)
        else:
            dataset_future = ts.open(spec, read=False, write=True)
        ts_dataset = dataset_future.result()
        return LazyNormalization(ts_dataset)
    except Exception as e:
        raise RuntimeError(f"Failed to open tensorstore array at {path}: {e}")


class LazyNormalization:
    """Wraps a tensorstore dataset, optionally normalizing on access."""

    def __init__(self, ts_dataset):
        self.ts_dataset = ts_dataset

    def __getitem__(self, index):
        result = self.ts_dataset[index]
        # Optionally: normalize result here if needed
        return result

    def __getattr__(self, attr):
        return getattr(self.ts_dataset, attr)


def exists(path):
    """Check if a file or S3 object exists. Handles S3 and local paths."""
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=True)
        s3_path = path.replace("zarr://", "")
        return fs.exists(s3_path)
    else:
        return os.path.exists(path)


def get_array_metadata(path):
    """Return metadata for a zarr or tensorstore array. Returns attrs as dict if available."""
    arr = open_zarr(path)
    if hasattr(arr, "attrs"):
        return arr.attrs.asdict()
    return {}
