import datetime
import os
from pathlib import Path
from collections.abc import Iterable
from typing import Callable

import blosc
import crc32c
import h5py
import hdf5plugin
import numpy as np
import numpy.typing as npt


def zarr3_shard_offsets_and_lengths(
    shard_file_name: str,
    shard_shape: tuple[int, ...] | None = None,
    chunk_shape: tuple[int, ...] | None = None,
    *,
    nchunks=None,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Extract chunk offsets and lengths from Zarr version 3 shard index

    Parameters
    ----------
    shard_file_name - string representing the path to the shard
    shard_shape - (Optional) tuple representing the size of the overall shard
    chunk_shape - (Optional) tuple representing the size of the chunks within the shard
    nchunks - (Optional) keyword argument to provide the number of chunks directly instead of the shape arguments
    """
    if nchunks is None:
        if len(shard_shape) != len(chunk_shape):
            raise Exception("Length of shard_shape should match length of chunk_shape")

        nchunks = 1
        for s, c in zip(shard_shape, chunk_shape):
            nchunks *= s // c

    index_length = nchunks * 16 + 4

    with open(shard_file_name, "rb") as f:
        f.seek(-index_length, 2)
        index_bytes = f.read()

    offsets_and_lengths_bytes = index_bytes[:-4]
    calc_checksum = crc32c.crc32c(offsets_and_lengths_bytes)
    index_checksum = int.from_bytes(index_bytes[-4:], "little")

    if calc_checksum != index_checksum:
        raise Exception("Calculated crc32c checksum does not match stored checksum")

    offsets_and_lengths = np.reshape(
        np.frombuffer(offsets_and_lengths_bytes, dtype="uint64", count=nchunks * 2),
        (nchunks, 2),
    )
    offsets = offsets_and_lengths[:, 0]
    lengths = offsets_and_lengths[:, 1]
    return offsets, lengths


def offsets_and_lengths_to_zarr3_trailer(
    offsets: npt.ArrayLike, lengths: npt.ArrayLike
) -> bytes:
    """
    Convert offsets and lengths 1-D arrays into Zarr 3 chunk index bytes

    Parameters
    ----------
    offsets: location in terms of bytes of the first byte of each chunk
    lengths: number of bytes within each chunk
    """
    offsets = offsets.astype("uint64")
    lengths = lengths.astype("uint64")
    offsets_and_lengths = np.transpose(np.vstack((offsets, lengths)))
    buffer = offsets_and_lengths.tobytes()
    checksum = crc32c.crc32c(buffer).to_bytes(4, "little")
    return buffer + checksum


def zarr3_prepend_shard(
    shard_file_name: str,
    shard_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    prefix: int | bytes,
):
    """
    Pad bytes to the beginning of an existing Zarr version 3 shard by shifting the chunks
    such that the first chunk starts after the given prefix.

    Parameters
    ----------
    shard_file_name - string representing the path to the shard
    shard_shape - tuple representing the size of the overall shard
    chunk_shape - tuple representing the size of the chunks within the shard
    prefix - either the number of bytes to use as a prefix or a bytes object to use as the prefix
    """
    offsets, lengths = zarr3_shard_offsets_and_lengths(
        shard_file_name, shard_shape, chunk_shape
    )

    nchunks = len(offsets)
    # 8 bytes for offset, 8 bytes for nbytes, and 4 bytes for checksum
    trailer_length = nchunks * 16 + 4

    with open(shard_file_name, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_length = f.tell()
        f.seek(0, os.SEEK_SET)
        shard_bytes = f.read(file_length - trailer_length)

    backup_shard_file_name = (
        shard_file_name + "_" + datetime.datetime.now().isoformat() + ".bak"
    )
    os.rename(shard_file_name, backup_shard_file_name)

    if type(prefix) is int:
        prefix = bytearray(prefix)

    offsets = np.array(offsets)
    offsets += len(prefix)

    trailer = offsets_and_lengths_to_zarr3_trailer(offsets, lengths)

    with open(shard_file_name, "wb") as f:
        f.write(prefix)
        f.write(shard_bytes)
        f.write(trailer)


def zarr3_shard_decompress_chunks(
    shard_file_name: str,
    shard_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    *,
    decompress: Callable[[bytes], bytes] = lambda x: x,
) -> list[npt.ArrayLike, ...]:
    """
    Decompress Zarr version 3 chunks from a shard.

    Return a list of decompressed chunks

    Parameters
    ----------
    shard_file_name - string representing the path to the shard
    shard_shape - tuple representing the size of the overall shard
    chunk_shape - tuple representing the size of the chunks within the shard
    dtype - data type
    decompress - (Optional) function that can be used to decompress a chunk. Defaults to raw.
                 Consider blosc.decompress
    """
    offsets, lengths = zarr3_shard_offsets_and_lengths(
        shard_file_name, shard_shape, chunk_shape
    )

    # decompress chunks
    chunks = []
    with open(shard_file_name, "rb") as f:
        for offset, length in zip(offsets, lengths):
            if offset == 0xFFFFFFFFFFFFFFFF:
                decompressed_chunk = np.zeros(chunk_shape)
            else:
                f.seek(offset, 0)
                compressed_chunk = f.read(length)
                decompressed_chunk = decompress(compressed_chunk)
                decompressed_chunk = np.reshape(
                    np.frombuffer(decompressed_chunk, dtype=dtype), chunk_shape
                )
            chunks.append(decompressed_chunk)
    return chunks


def zarr3_shard_raw_chunks(
    shard_file_name: str,
    shard_shape: tuple[int, ...] = None,
    chunk_shape: tuple[int, ...] = None,
    *,
    offsets: npt.ArrayLike | None = None,
    lengths: npt.ArrayLike | None = None,
    nchunks: int | None = None,
) -> list[bytes, ...]:
    """
    Return a list of raw chunks from a Zarr version 3 shard

    Parameters
    ----------
    shard_file_name - string representing the path to the shard
    shard_shape - (Optional) tuple representing the size of the overall shard
    chunk_shape - (Optional) tuple representing the size of the chunks within the shard
    offsets - (Optional) array of byte locations of the chunks, can be provided instead of shape information
    lengths - (Optional) array of the number of bytes contained within each chunk, can be provided instead of shape information
    nchunks - (Optional) number of chunks that can be provided if known

    One of the following sets of parameters are required:
    1. shard_shape and chunk_shape
    2. offsets and lengths
    """
    if offsets is None or lengths is None:
        offsets, lengths = zarr3_shard_offsets_and_lengths(
            shard_file_name, shard_shape, chunk_shape, nchunks=nchunks
        )

    # read raw chunks
    raw_chunks = []
    with open(shard_file_name, "rb") as f:
        for offset, length in zip(offsets, lengths):
            if offset == 0xFFFFFFFFFFFFFFFF:
                raw_chunks.append(b"")
            else:
                f.seek(offset, 0)
                compressed_chunk = f.read(length)
                raw_chunks.append(compressed_chunk)
    return raw_chunks


def min_meta_block_size(nchunks: int) -> int:
    """
    Calculate the minimum meta block size for a HDF5 file
    to hold a chunked dataset for the shard data and
    a contiguous dataset for the shard index

    Parameters
    ----------
    nchunks - number of chunks
    """
    # To ensure HDF5 metadata is consolidated at the beginning
    # set the meta_block_size to be large enough to hold all
    # the metadata. The non-chunk metadata requires about 736 bytes.
    # We use 1024 as the next power of 2 greater than 736.
    meta_block_size = max(1024 + nchunks * 16, nchunks * 32)

    # For neatness, keep meta_block_size to be a power of 2
    meta_block_size = 2 ** np.ceil(np.log2(meta_block_size)).astype("int")

    # The default minimum meta_block_size is 2048 bytes
    return meta_block_size


def zarr3_shard_decompress(
    shard_file_name: str,
    shard_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    dtype: str,
) -> npt.ArrayLike:
    """
    Decompress the contents of a Zarr version 3 shard.

    Return a numpy ndarray as the size of the entire shard

    Parameters
    ----------
    shard_file_name - string representing the path to the shard
    shard_shape - tuple representing the size of the overall shard
    chunk_shape - tuple representing the size of the chunks within the shard
    dtype - data type of the chunks
    """
    # list of chunks in lexicographical order
    chunks = zarr3_shard_decompress_chunks(
        shard_file_name,
        shard_shape,
        chunk_shape,
        dtype,
    )

    # chunk count per axis
    chunk_count = tuple(map(lambda s, c: s // c, shard_shape, chunk_shape))

    # initial shard has shape
    # (chunk_count[0], chunk_count[1], ..., chunk_shape[0], chunk_shape[1], ...)
    shard = np.reshape(np.stack(chunks), chunk_count + chunk_shape)

    # pemute axes to obtain
    # (chunk_count[0], chunk_shape[0], chunk_count[1], chunk_shape[1], ...)
    permute_order = tuple(
        np.transpose(np.reshape(range(shard.ndim), (2, len(chunk_shape)))).flat
    )
    shard = np.transpose(shard, axes=permute_order)

    shard = np.reshape(shard, shard_shape)
    return shard


def zarr3_shard_hdf5_template(
    shard_file_name: str,
    shard_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    compression: dict = {},
    *,
    verify: bool = False,
    delete_backup: bool = False,
    raw_chunks: list[bytes] | None = None,
    chunk_write_order: Iterable[int] | None = None,
) -> None:
    """
    Create a hybrid HDF5-Zarr version 3 shard from an existing Zarr version 3 shard

    Parameters
    ----------
    shard_file_name:    Path to shard file
    shard_shape:        Tuple describing dimensions of a shard
    chunk_shape:        Tuple describing dimensions of a chunk
    dtype:              Data type of the array
    compression:        (Optional) Default: {}. Expanded to keywords for h5py.create_dataset.
    verify:             (Optional) Default: False. If True, compare decompressed shard with backup shard.
    delete_backup:      (Optional) Default: False. If True and verify is True, delete backup file if verified.
    raw_chunks:         (Optional) Default: None. Use given raw chunks instead of reading from an existing shard.
    chunk_write_order:  (Optional) Default: None. Order to write chunks in.
    """
    if len(shard_shape) != len(chunk_shape):
        raise Exception("Length of shard_shape and chunk_shape do not match")

    if raw_chunks is None:
        # Read information from shard
        offsets, lengths = zarr3_shard_offsets_and_lengths(
            shard_file_name, shard_shape, chunk_shape
        )
        raw_chunks = zarr3_shard_raw_chunks(
            shard_file_name, offsets=offsets, lengths=lengths
        )
        if chunk_write_order is None:
            chunk_write_order = np.argsort(offsets)
    else:
        offsets = np.zeros(len(raw_chunks), dtype="uint64")

    if chunk_write_order is None:
        chunk_write_order = range(len(raw_chunks))

    # Backup shard
    backup_shard_file_name = (
        shard_file_name + "_" + datetime.datetime.now().isoformat() + ".bak"
    )
    if Path(shard_file_name).is_file():
        os.rename(shard_file_name, backup_shard_file_name)

    # number of chunks per axes
    nc = tuple(map(lambda s, c: s // c, shard_shape, chunk_shape))
    nchunks = np.prod(nc)

    with h5py.File(
        shard_file_name,
        "w",
        meta_block_size=min_meta_block_size(nchunks),
        libver="v110",
    ) as h5f:
        # store the shard data into a chunked dataset
        h5ds = h5f.create_dataset(
            "zarrshard", shard_shape, dtype=dtype, chunks=chunk_shape, **compression
        )
        coordinates = [idx for idx in np.ndindex(nc)]
        for i in chunk_write_order:
            if offsets[i] != 0xFFFFFFFFFFFFFFFF:
                h5ds.id.write_direct_chunk(
                    tuple(map(lambda x, c: x * c, coordinates[i], chunk_shape)),
                    raw_chunks[i],
                )

        # collect chunk byte offsets and sizes
        chunk_info = {}
        h5ds.id.chunk_iter(lambda x: chunk_info.__setitem__(x.chunk_offset, x))
        offsets_and_lengths = np.empty((nchunks, 2), dtype="uint64")
        for i, coords in enumerate(coordinates):
            chunk_offset = tuple(map(lambda x, c: x * c, coords, chunk_shape))
            ci = chunk_info.get(chunk_offset, None)
            if ci is None:
                # missing chunk
                offsets_and_lengths[i, 0] = 0xFFFFFFFFFFFFFFFF
                offsets_and_lengths[i, 1] = 0xFFFFFFFFFFFFFFFF
            else:
                offsets_and_lengths[i, 0] = ci.byte_offset
                offsets_and_lengths[i, 1] = ci.size
        offsets_and_lengths_bytes = offsets_and_lengths.tobytes()

        # build Zarr3 shard index as a HDF5 byte dataset
        index_h5ds = h5f.create_dataset(
            "zarrindex", (len(offsets_and_lengths_bytes) + 4,), dtype="uint8"
        )
        index_h5ds[:-4] = np.frombuffer(offsets_and_lengths_bytes, dtype="uint8")
        index_h5ds[-4:] = np.frombuffer(
            crc32c.crc32c(offsets_and_lengths_bytes).to_bytes(4, "little"),
            dtype="uint8",
        )

    if verify:
        if Path(backup_shard_file_name).is_file():
            a = zarr3_shard_decompress(shard_file_name, shard_shape, chunk_shape, dtype)
            b = zarr3_shard_decompress(
                backup_shard_file_name, shard_shape, chunk_shape, dtype
            )
            if not np.array_equal(a, b):
                raise Exception("Unable to verify shard via decompression")
            if delete_backup:
                os.remove(backup_shard_file_name)
        else:
            raise Exception("Unable to verify shard since backup file does not exist")

    return None
