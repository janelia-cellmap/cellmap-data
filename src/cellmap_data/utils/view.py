import time
import webbrowser
import neuroglancer
from IPython.display import IFrame, display
from IPython import get_ipython
import numpy as np
from upath import UPath
import zarr
from multiprocessing.pool import ThreadPool


def open_neuroglancer(metadata):
    from cellmap_flow.utils.scale_pyramid import ScalePyramid
    from cellmap_flow.utils.ds import open_ds_tensorstore

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
    import xarray_tensorstore as xt
    import tensorstore

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
