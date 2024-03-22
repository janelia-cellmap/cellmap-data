from typing import Union, Iterable, Callable

import math
import zarr
import numpy as np

from zarrdataset._utils import (
    map_axes_order,
    scale_coords,
    select_axes,
    parse_rois,
    translate2roi,
)
from zarrdataset import ImageLoader, image2array


class CellMapImageLoader(ImageLoader):
    """Image lazy loader class.

    Opens the zarr file, or any image that can be open by TiffFile or PIL, as a
    Zarr array.

    Parameters
    ----------
    filename: str
    source_axes: str
    data_group: Union[str, None]
    axes: Union[str, None]
    roi: Union[str, slice, Iterable[slice], None]
    image_func: Union[Callable, None]
    zarr_store: Union[zarr.storage.Store, None]
    spatial_axes: str
    mode: str
    """

    def __init__(
        self,
        filename: str,
        source_axes: str,
        data_group: Union[str, None] = None,
        axes: Union[str, None] = None,
        roi: Union[str, slice, Iterable[slice], None] = None,
        image_func: Union[Callable, None] = None,
        zarr_store: Union[zarr.storage.Store, None] = None,
        spatial_axes: str = "ZYX",
        mode: str = "",
    ):
        """Class extending the ImageLoader class to leverage CellMap metadata and file structure"""
        self.mode = mode
        self.spatial_axes = spatial_axes

        if roi is None:
            parsed_roi = [slice(None)] * len(source_axes)
        elif isinstance(roi, str):
            parsed_roi = parse_rois([roi])[0]
        elif isinstance(roi, (list, tuple)):
            if len(roi) != len(source_axes):
                raise ValueError(
                    f"ROIs does not match the number of the array"
                    f" axes. Expected {len(source_axes)}, got "
                    f"{len(roi)}"
                )
            elif not all([isinstance(roi_ax, slice) for roi_ax in roi]):
                raise ValueError(
                    f"ROIs must be slices, but got "
                    f"{[type(roi_ax) for roi_ax in roi]}"
                )
            else:
                parsed_roi = roi
        elif isinstance(roi, slice):
            if len(source_axes) > 1 and not (roi.start is None and roi.stop is None):
                raise ValueError(
                    f"ROIs must specify a slice per axes. "
                    f"Expected {len(source_axes)} slices, got "
                    f"only {roi}"
                )
            else:
                parsed_roi = [roi] * len(source_axes)
        else:
            raise ValueError(
                f"Incorrect ROI format, expected a list of "
                f"slices, or a parsable string, got {roi}"
            )

        roi_slices = list(
            map(
                lambda r: slice(r.start if r.start is not None else 0, r.stop, None),
                parsed_roi,
            )
        )

        (self.arr, self._store) = image2array(
            filename, data_group=data_group, zarr_store=zarr_store
        )

        self.roi = roi_slices
        self.source_axes = source_axes
        self.axes = source_axes

        if axes is not None and axes != source_axes:
            source_axes_list = list(source_axes)
            self._drop_axes = list(set(source_axes) - set(axes))
            for d_a in sorted(
                (source_axes_list.index(a) for a in self._drop_axes), reverse=True
            ):
                if self.roi[d_a].stop is not None:
                    roi_len = self.roi[d_a].stop
                else:
                    roi_len = self.arr.shape[d_a]

                roi_len -= self.roi[d_a].start

                if roi_len > 1:
                    raise ValueError(
                        f"Cannot drop axis `{source_axes[d_a]}` "
                        f"(from `source_axes={source_axes}`) "
                        f"because no selection was made for it, "
                        f"and it is not being used as image axes "
                        f"thereafter (`axes={axes}`)."
                    )

            self.permute_order = map_axes_order(source_axes, axes)
            self._new_axes = list(set(axes) - set(source_axes))
            self.axes = axes
        else:
            self.permute_order = list(range(len(self.axes)))

        self._image_func = image_func

        if image_func is not None:
            self.axes = image_func.axes

    def __del__(self):
        # Close the connection to the image file
        if self._store is not None:
            self._store.close()


class ImageCollection(object):
    """A class to contain a collection of inputs from different modalities.

    This is used to match images with their respective labels and masks.

    Parameters
    ----------
    collection_args : dict
        Collection arguments containing specifications to open `images`,
        `masks`, `labels`, etc.
    spatial_axes : str
        Spatial axes of the dataset, which are used to match different
        modalities using as reference these axes from the `images` collection.
    """

    def __init__(self, collection_args: dict, spatial_axes: str = "ZYX"):

        self.reference_mode = list(collection_args.keys())[0]

        self.spatial_axes = spatial_axes

        self.collection = dict(
            (
                (mode, ImageLoader(spatial_axes=spatial_axes, mode=mode, **mode_args))
                for mode, mode_args in collection_args.items()
            )
        )

        self._generate_mask()
        self.reset_scales()

    def _generate_mask(self):
        mask_modes = list(filter(lambda k: "mask" in k, self.collection.keys()))

        if len(mask_modes) > 0:
            self.mask_mode = mask_modes[0]
            return

        ref_axes = self.collection[self.reference_mode].axes
        ref_shape = self.collection[self.reference_mode].shape
        ref_chunk_size = self.collection[self.reference_mode].chunk_size

        mask_axes = list(set(self.spatial_axes).intersection(ref_axes))
        mask_axes_ord = map_axes_order(mask_axes, ref_axes)
        mask_axes = [mask_axes[a] for a in mask_axes_ord]

        mask_chunk_size = [
            int(math.ceil(s / c))
            for s, c, a in zip(ref_shape, ref_chunk_size, ref_axes)
            if a in mask_axes
        ]

        self.collection["masks"] = ImageBase(
            shape=mask_chunk_size,
            chunk_size=mask_chunk_size,
            source_axes=mask_axes,
            mode="masks",
        )
        self.mask_mode = "masks"

    def reset_scales(self) -> None:
        """Reset the scales between data modalities to match the `images`
        collection shape on the `spatial_axes` only.
        """
        img_shape = self.collection[self.reference_mode].shape
        img_source_axes = self.collection[self.reference_mode].source_axes
        img_axes = self.collection[self.reference_mode].axes

        spatial_reference_axes = [ax for ax in img_axes if ax in self.spatial_axes]
        spatial_reference_shape = [
            img_shape[img_axes.index(ax)] if ax in img_source_axes else 1
            for ax in spatial_reference_axes
        ]

        for img in self.collection.values():
            img.rescale(spatial_reference_shape, spatial_reference_axes)

    def __getitem__(self, index):
        collection_set = dict(
            (mode, img[index]) for mode, img in self.collection.items()
        )

        return collection_set
