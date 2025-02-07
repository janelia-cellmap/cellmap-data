import json
from typing import Optional

import zarr


def generate_base_multiscales_metadata(
    ds_name: str,
    scale_level: int,
    voxel_size: list,
    translation: list,
    units: str,
    axes: list,
):
    if ".zarr" in ds_name:
        ds_name = ds_name.split(".zarr")[-1]
    z_attrs: dict = {"multiscales": [{}]}
    z_attrs["multiscales"][0]["axes"] = [
        {"name": axis, "type": "space", "unit": units} for axis in axes
    ]
    z_attrs["multiscales"][0]["coordinateTransformations"] = [
        {"scale": [1.0, 1.0, 1.0], "type": "scale"}
    ]
    z_attrs["multiscales"][0]["datasets"] = [
        {
            "coordinateTransformations": [
                {"scale": voxel_size, "type": "scale"},
                {"translation": translation, "type": "translation"},
            ],
            "path": f"s{scale_level}",
        }
    ]

    z_attrs["multiscales"][0]["name"] = ds_name
    z_attrs["multiscales"][0]["version"] = "0.4"

    return z_attrs


def add_multiscale_metadata_levels(multsc, base_scale_level, levels_to_add):
    # store original array in a new .zarr file as an arr_name scale
    z_attrs = multsc
    # print(z_attrs)
    base_scale = z_attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        0
    ]["scale"]
    base_trans = z_attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        1
    ]["translation"]
    for level in range(base_scale_level, base_scale_level + levels_to_add):
        # print(f"{level=}")

        # break the slices up into batches, to make things easier for the dask scheduler
        sn = [dim * pow(2, level) for dim in base_scale]
        trn = [
            (dim * (pow(2, level - 1) - 0.5)) + tr
            for (dim, tr) in zip(base_scale, base_trans)
        ]

        z_attrs["multiscales"][0]["datasets"].append(
            {
                "coordinateTransformations": [
                    {"type": "scale", "scale": sn},
                    {"type": "translation", "translation": trn},
                ],
                "path": f"s{level + 1}",
            }
        )

    return z_attrs


def create_multiscale_metadata(
    ds_name: str,
    voxel_size: list,
    translation: list,
    units: str,
    axes: list,
    base_scale_level: int = 0,
    levels_to_add: int = 0,
    out_path: Optional[str] = None,
):
    z_attrs = generate_base_multiscales_metadata(
        ds_name, base_scale_level, voxel_size, translation, units, axes
    )
    if levels_to_add > 0:
        z_attrs = add_multiscale_metadata_levels(
            z_attrs, base_scale_level, levels_to_add
        )

    if out_path is not None:
        write_metadata(z_attrs, out_path)
    else:
        return z_attrs


def write_metadata(z_attrs, out_path):
    with open(out_path, "w") as f:
        f.write(json.dumps(z_attrs, indent=4))


def find_level(path: str, target_scale: dict[str, float]) -> str:
    """Finds the multiscale level that is closest to the target scale."""
    group = zarr.open(path, mode="r")
    # Get the order of axes in the image
    axes = []
    for axis in group.attrs["multiscales"][0]["axes"]:
        if axis["type"] == "space":
            axes.append(axis["name"])

    last_path: str | None = None
    scale = {}
    for level in group.attrs["multiscales"][0]["datasets"]:
        for transform in level["coordinateTransformations"]:
            if "scale" in transform:
                scale = {c: s for c, s in zip(axes, transform["scale"])}
                break
        for c in axes:
            if scale[c] > target_scale[c]:
                if last_path is None:
                    return level["path"]
                else:
                    return last_path
        last_path = level["path"]
    return last_path  # type: ignore
