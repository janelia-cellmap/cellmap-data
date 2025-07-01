import json
import logging
import operator
import os
import re
import time
import webbrowser
from multiprocessing.pool import ThreadPool

import neuroglancer
import numpy as np
import urllib
import s3fs
import zarr
import tensorstore as ts

from IPython import get_ipython
from IPython.display import IFrame, display
from upath import UPath

logger = logging.getLogger(__name__)

# S3 bucket names and paths for Janelia COSEM datasets
GT_S3_BUCKET = "janelia-cosem-datasets"
RAW_S3_BUCKET = "janelia-cosem-datasets"
S3_SEARCH_PATH = "{dataset}/{dataset}.zarr/recon-1/{name}"
S3_CROP_NAME = "labels/groundtruth/{crop}/{label}"
S3_RAW_NAME = "em/fibsem-uint8"


def get_multiscale_voxel_sizes(path: str):
    if "s3://" in path:
        # Use s3fs to read the zarr metadata
        fs = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(
            root=path.removeprefix("zarr://s3://"),
            s3=fs,
            check=False,  # skip consistency checks for speed
        )
        ds = zarr.open(store, mode="r")
    else:
        # Use local zarr store
        ds = zarr.open(path, mode="r")
    voxel_sizes = {}
    for scale_ds in ds.attrs["multiscales"][0]["datasets"]:
        for transform in scale_ds["coordinateTransformations"]:
            if transform["type"] == "scale":
                voxel_sizes[scale_ds["path"]] = transform["scale"]
                break
    if not voxel_sizes:
        raise ValueError(
            f"No scale transformations found in the zarr metadata at {path}"
        )
    return voxel_sizes


def get_neuroglancer_link(metadata):
    # extract dataset name from raw_path
    m = re.search(r"/([^/]+)/\\1\\.zarr", metadata["raw_path"])
    if m:
        dataset = m.group(1)
    else:
        # fallback: take parent folder name before .zarr
        import os

        dataset = os.path.basename(metadata["raw_path"].split(".zarr")[0])
    # build raw EM layer source
    raw_key = S3_SEARCH_PATH.format(dataset=dataset, name=S3_RAW_NAME)
    raw_source = f"zarr://s3://{RAW_S3_BUCKET}/{raw_key}"
    voxel_sizes = [get_multiscale_voxel_sizes(raw_source)]
    layers = {"raw": {"type": "image", "source": raw_source}}
    # segmentation layers
    # extract crop identifier from target_path_str
    m2 = re.search(r"labels/groundtruth/([^/]+)/", metadata["target_path_str"])
    crop = m2.group(1) if m2 else ""
    for class_name in metadata["class_weights"].keys():
        seg_path = S3_CROP_NAME.format(crop=crop, label=class_name)
        seg_key = S3_SEARCH_PATH.format(dataset=dataset, name=seg_path)
        seg_source = f"zarr://s3://{GT_S3_BUCKET}/{seg_key}"
        # get voxel size for this segmentation layer
        voxel_sizes.append(get_multiscale_voxel_sizes(seg_source))
        layers[class_name] = {"type": "segmentation", "source": seg_source}

    # find the minimum voxel size across all layers
    voxel_size = np.min(
        [
            np.min([np.array(vs) for vs in ds_vs.values()], axis=0)
            for ds_vs in voxel_sizes
        ],
        axis=0,
    ).tolist()
    # navigation pose (x, y, z)
    position = [
        metadata["current_center"]["z"] / voxel_size[0],
        metadata["current_center"]["y"] / voxel_size[1],
        metadata["current_center"]["x"] / voxel_size[2],
    ]
    state = {
        "layers": layers,
        "navigation": {"pose": {"position": {"voxelCoordinates": position}}},
    }
    fragment = urllib.parse.quote(json.dumps(state), safe='/:,"{}[]')
    return f"https://neuroglancer-demo.appspot.com/#!{fragment}"


def open_neuroglancer(metadata):
    """
    Launch a Neuroglancer viewer showing raw data and labels,
    centered on the point in metadata['current_center'].

    metadata: dict with keys
      - 'raw_path': path to your raw Zarr (no .zarr extension in source)
      - 'current_center': {'x':…, 'y':…, 'z':…}
      - 'target_path_str': format string with '{label}' for each class
      - 'class_weights': dict mapping class names to weights

    Returns the Neuroglancer.Viewer object.
    """
    # 1) bind to localhost on a random port
    neuroglancer.set_server_bind_address("localhost", 0)
    viewer = neuroglancer.Viewer()

    # 2) build layer sources
    raw_source = get_layer(
        metadata["raw_path"],
        layer_type="image",
    )
    label_layers = {}
    for class_name in metadata["class_weights"].keys():
        # fill in the placeholder
        label_path = metadata["target_path_str"].format(label=class_name)
        label_layers[class_name] = get_layer(
            label_path,
            layer_type="segmentation",
        )

    # 3) push state in one atomic txn
    with viewer.txn() as s:
        # raw intensity volume
        s.layers["raw"] = raw_source
        # one layer per class
        for class_name, layer in label_layers.items():
            s.layers[class_name] = layer

    # 4) display inline or print URL
    url = viewer.get_viewer_url()
    print(f"Neuroglancer viewer URL: {url}")
    if get_ipython() is not None:
        # If running in Jupyter, display the viewer inline
        viewer_iframe = IFrame(url, width=1000, height=600)
        display(viewer_iframe)
    else:
        webbrowser.open(url)

    # 5) center the view on the current center when it is available by starting a background thread
    def _center_view():
        while len(viewer.state.dimensions.to_json()) < 3:
            time.sleep(0.1)  # wait for dimensions to be set
        with viewer.txn() as s:
            # jump to the stored center (x, y, z)
            cx = float(metadata["current_center"]["x"]) / (
                viewer.state.dimensions["x"].scale * 10**9
            )
            cy = float(metadata["current_center"]["y"]) / (
                viewer.state.dimensions["y"].scale * 10**9
            )
            cz = float(metadata["current_center"]["z"]) / (
                viewer.state.dimensions["z"].scale * 10**9
            )
            # (z is the first dimension in Neuroglancer)
            s.position = [cz, cy, cx]

    pool = ThreadPool(processes=1)
    pool.apply_async(_center_view)
    return viewer


def get_layer(
    data_path: str,
    layer_type: str = "image",
    multiscale: bool = True,
) -> neuroglancer.Layer:
    """
    Get a Neuroglancer layer from a zarr data path for a LocalVolume.

    Parameters
    ----------
    data_path : str
        The path to the zarr data.
    layer_type : str
        The type of layer to get. Can be "image" or "segmentation". Default is "image".
    multiscale : bool
        Whether the metadata is OME-NGFF multiscale. Default is True.

    Returns
    -------
    neuroglancer.Layer
        The Neuroglancer layer.
    """
    # Construct an xarray with Tensorstore backend
    # Get metadata
    if multiscale:
        # Add all scales
        layers = []
        scales, metadata = parse_multiscale_metadata(data_path)
        for scale in scales:
            this_path = (UPath(data_path) / scale).path
            image = open_ds_tensorstore(this_path)
            # image = get_image(this_path)

            layers.append(
                neuroglancer.LocalVolume(
                    data=image,
                    dimensions=neuroglancer.CoordinateSpace(
                        scales=metadata[scale]["voxel_size"],
                        units=metadata[scale]["units"],
                        names=metadata[scale]["names"],
                    ),
                    voxel_offset=metadata[scale]["voxel_offset"],
                )
            )
        volume = ScalePyramid(layers)

    else:
        # Handle single scale zarr files
        names = ["z", "y", "x"]
        units = ["nm", "nm", "nm"]
        attrs = zarr.open(data_path, mode="r").attrs.asdict()
        if "voxel_size" in attrs:
            voxel_size = attrs["voxel_size"]
        elif "resolution" in attrs:
            voxel_size = attrs["resolution"]
        elif "scale" in attrs:
            voxel_size = attrs["scale"]
        else:
            voxel_size = [1, 1, 1]

        if "translation" in attrs:
            translation = attrs["translation"]
        elif "offset" in attrs:
            translation = attrs["offset"]
        else:
            translation = [0, 0, 0]

        voxel_offset = np.array(translation) / np.array(voxel_size)

        image = open_ds_tensorstore(data_path)
        # image = get_image(data_path)

        volume = neuroglancer.LocalVolume(
            data=image,
            dimensions=neuroglancer.CoordinateSpace(
                scales=voxel_size,
                units=units,
                names=names,
            ),
            voxel_offset=voxel_offset,
        )

    if layer_type == "segmentation":
        return neuroglancer.SegmentationLayer(source=volume)
    else:
        return neuroglancer.ImageLayer(source=volume)


def get_image(data_path: str):
    import tensorstore
    import xarray_tensorstore as xt

    try:
        return open_ds_tensorstore(data_path)
    except ValueError as e:
        spec = xt._zarr_spec_from_path(data_path)
        array_future = tensorstore.open(spec, read=True, write=False)
        try:
            array = array_future.result()
        except ValueError as e:
            Warning(e)
            UserWarning("Falling back to zarr3 driver")
            spec["driver"] = "zarr3"
            array_future = tensorstore.open(spec, read=True, write=False)
            array = array_future.result()
        return array


def parse_multiscale_metadata(data_path: str):
    metadata = zarr.open(data_path, mode="r").attrs.asdict()["multiscales"][0]
    scales = []
    parsed = {}
    for ds in metadata["datasets"]:
        scales.append(ds["path"])

        names = []
        units = []
        translation = []
        voxel_size = []

        for axis in metadata["axes"]:
            if axis["name"] == "c":
                names.append("c^")
                voxel_size.append(1)
                translation.append(0)
                units.append("")
            else:
                names.append(axis["name"])
                units.append("nm")

        for transform in ds["coordinateTransformations"]:
            if transform["type"] == "scale":
                voxel_size.extend(transform["scale"])
            elif transform["type"] == "translation":
                translation.extend(transform["translation"])

        parsed[ds["path"]] = {
            "names": names,
            "units": units,
            "voxel_size": voxel_size,
            "translation": translation,
            "voxel_offset": np.array(translation) / np.array(voxel_size),
        }
    scales.sort(key=lambda x: int(x[1:]))
    return scales, parsed


class ScalePyramid(neuroglancer.LocalVolume):
    """A neuroglancer layer that provides volume data on different scales.
    Mimics a LocalVolume.
    From https://github.com/funkelab/funlib.show.neuroglancer/blob/master/funlib/show/neuroglancer/scale_pyramid.py

    Args:

            volume_layers (``list`` of ``LocalVolume``):

                One ``LocalVolume`` per provided resolution.
    """

    def __init__(self, volume_layers):
        volume_layers = volume_layers

        super(neuroglancer.LocalVolume, self).__init__()

        logger.debug("Creating scale pyramid...")

        self.min_voxel_size = min(
            [tuple(layer.dimensions.scales) for layer in volume_layers]
        )
        self.max_voxel_size = max(
            [tuple(layer.dimensions.scales) for layer in volume_layers]
        )

        self.dims = len(volume_layers[0].dimensions.scales)
        self.volume_layers = {
            tuple(
                int(x)
                for x in map(
                    operator.truediv, layer.dimensions.scales, self.min_voxel_size
                )
            ): layer
            for layer in volume_layers
        }

        logger.debug("min_voxel_size: %s", self.min_voxel_size)
        logger.debug("scale keys: %s", self.volume_layers.keys())
        logger.debug(self.info())

    @property
    def volume_type(self):
        return self.volume_layers[(1,) * self.dims].volume_type

    @property
    def token(self):
        return self.volume_layers[(1,) * self.dims].token

    def info(self):
        reference_layer = self.volume_layers[(1,) * self.dims]
        # return reference_layer.info()

        reference_info = reference_layer.info()

        info = {
            "dataType": reference_info["dataType"],
            "encoding": reference_info["encoding"],
            "generation": reference_info["generation"],
            "coordinateSpace": reference_info["coordinateSpace"],
            "shape": reference_info["shape"],
            "volumeType": reference_info["volumeType"],
            "voxelOffset": reference_info["voxelOffset"],
            "chunkLayout": reference_info["chunkLayout"],
            "downsamplingLayout": reference_info["downsamplingLayout"],
            "maxDownsampling": int(
                np.prod(np.array(self.max_voxel_size) // np.array(self.min_voxel_size))
            ),
            "maxDownsampledSize": reference_info["maxDownsampledSize"],
            "maxDownsamplingScales": reference_info["maxDownsamplingScales"],
        }

        return info

    def get_encoded_subvolume(self, data_format, start, end, scale_key=None):
        if scale_key is None:
            scale_key = ",".join(("1",) * self.dims)

        scale = tuple(int(s) for s in scale_key.split(","))
        closest_scale = None
        min_diff = np.inf
        for volume_scales in self.volume_layers.keys():
            scale_diff = np.array(scale) // np.array(volume_scales)
            if any(scale_diff < 1):
                continue
            scale_diff = scale_diff.max()
            if scale_diff < min_diff:
                min_diff = scale_diff
                closest_scale = volume_scales

        assert closest_scale is not None
        relative_scale = np.array(scale) // np.array(closest_scale)

        return self.volume_layers[closest_scale].get_encoded_subvolume(
            data_format, start, end, scale_key=",".join(map(str, relative_scale))
        )

    def get_object_mesh(self, object_id):
        return self.volume_layers[(1,) * self.dims].get_object_mesh(object_id)

    def invalidate(self):
        return self.volume_layers[(1,) * self.dims].invalidate()


def open_ds_tensorstore(dataset_path: str, mode="r", concurrency_limit=None):
    # open with zarr or n5 depending on extension
    filetype = (
        "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
    )
    extra_args = {}

    if dataset_path.startswith("s3://"):
        kvstore = {
            "driver": "s3",
            "bucket": dataset_path.split("/")[2],
            "path": "/".join(dataset_path.split("/")[3:]),
            "aws_credentials": {
                "anonymous": True,
            },
        }
    elif dataset_path.startswith("gs://"):
        # check if path ends with s#int
        if ends_with_scale(dataset_path):
            scale_index = int(dataset_path.rsplit("/s")[1])
            dataset_path = dataset_path.rsplit("/s")[0]
        else:
            scale_index = 0
        filetype = "neuroglancer_precomputed"
        kvstore = dataset_path
        extra_args = {"scale_index": scale_index}
    else:
        kvstore = {
            "driver": "file",
            "path": os.path.normpath(dataset_path),
        }

    if concurrency_limit:
        spec = {
            "driver": filetype,
            "context": {
                "data_copy_concurrency": {"limit": concurrency_limit},
                "file_io_concurrency": {"limit": concurrency_limit},
            },
            "kvstore": kvstore,
            **extra_args,
        }
    else:
        spec = {"driver": filetype, "kvstore": kvstore, **extra_args}

    if mode == "r":
        dataset_future = ts.open(spec, read=True, write=False)
    else:
        dataset_future = ts.open(spec, read=False, write=True)

    if dataset_path.startswith("gs://"):
        # NOTE: Currently a hack since google store is for some reason stored as mutlichannel
        ts_dataset = dataset_future.result()[ts.d["channel"][0]]
    else:
        ts_dataset = dataset_future.result()

    # return ts_dataset
    return LazyNormalization(ts_dataset)


def ends_with_scale(string):
    pattern = (
        r"s\d+$"  # Matches 's' followed by one or more digits at the end of the string
    )
    return bool(re.search(pattern, string))


class LazyNormalization:
    def __init__(self, ts_dataset):
        self.ts_dataset = ts_dataset

    def __getitem__(self, index):
        result = self.ts_dataset[index]
        return apply_norms(result)

    def __getattr__(self, attr):
        at = getattr(self.ts_dataset, attr)
        if attr == "dtype":
            if len(g.input_norms) > 0:
                return np.dtype(g.input_norms[-1].dtype)
            return np.dtype(at.numpy_dtype)
        return at


def apply_norms(data):
    if hasattr(data, "read"):
        data = data.read().result()
    for norm in g.input_norms:
        data = norm(data)
    return data
