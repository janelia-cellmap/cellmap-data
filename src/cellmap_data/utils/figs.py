import io
from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_image_grid(
    input_data: torch.Tensor,
    target_data: torch.Tensor,
    outputs: torch.Tensor,
    classes: Sequence[str],
    batch_size: Optional[int] = None,
    fig_size: int = 3,
    clim: Optional[Sequence] = None,
    cmap: Optional[str] = None,
) -> plt.Figure:  # type: ignore
    """
    Create a grid of images for input, target, and output data.
    Args:
        input_data (torch.Tensor): Input data.
        target_data (torch.Tensor): Target data.
        outputs (torch.Tensor): Model outputs.
        classes (list): List of class labels.
        batch_size (int, optional): Number of images to display. Defaults to the length of the first axis of 'input_data'.
        fig_size (int, optional): Size of the figure. Defaults to 3.
        clim (tuple, optional): Color limits for the images. Defaults to be scaled by the image's intensity.
        cmap (str, optional): Colormap for the images. Defaults to None.

    Returns:
        fig (matplotlib.figure.Figure): Figure object.
    """
    if batch_size is None:
        batch_size = input_data.shape[0]
    num_images = len(classes) * 2 + 2
    fig, ax = plt.subplots(
        batch_size, num_images, figsize=(fig_size * num_images, fig_size * batch_size)
    )
    if len(ax.shape) == 1:
        ax = ax[None, :]
    for b in range(batch_size):
        for c, label in enumerate(classes):
            output = outputs[b][c].squeeze().cpu().detach().numpy()
            target = target_data[b][c].squeeze().cpu().detach().numpy()
            if len(output.shape) == 3:
                output_mid = output.shape[0] // 2
                output = output[output_mid]
                target = target[output_mid]
            ax[b, c * 2 + 2].imshow(target, clim=clim, cmap=cmap)
            ax[b, c * 2 + 2].axis("off")
            ax[b, c * 2 + 2].set_title(f"GT {label}")
            ax[b, c * 2 + 3].imshow(output, clim=clim, cmap=cmap)
            ax[b, c * 2 + 3].axis("off")
            ax[b, c * 2 + 3].set_title(f"Pred. {label}")
        input_img = input_data[b][0].squeeze().cpu().detach().numpy()
        if len(input_img.shape) == 3:
            input_mid = input_img.shape[0] // 2
            input_img = input_img[input_mid]
        x_pad, y_pad = (input_img.shape[1] - output.shape[1]) // 2, (
            input_img.shape[0] - output.shape[0]
        ) // 2
        if x_pad <= 0:
            x_slice = slice(0, input_img.shape[1])
        else:
            x_slice = slice(x_pad, -x_pad)
        if y_pad <= 0:
            y_slice = slice(0, input_img.shape[0])
        else:
            y_slice = slice(y_pad, -y_pad)
        ax[b, 1].imshow(input_img[x_slice, y_slice], cmap="gray", clim=clim)
        ax[b, 1].axis("off")
        ax[b, 1].set_title("Raw")
        ax[b, 0].imshow(input_img, cmap="gray", clim=clim)
        ax[b, 0].axis("off")
        ax[b, 0].set_title("Full FOV")
        w, h = output.shape[1], output.shape[0]
        rect = plt.Rectangle(
            (x_pad, y_pad), w, h, edgecolor="r", facecolor="none"
        )  # type: ignore
        ax[b, 0].add_patch(rect)
    fig.tight_layout()
    return fig


def fig_to_image(fig: plt.Figure) -> np.ndarray:  # type: ignore
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw", dpi=fig.dpi)
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    plt.close("all")
    return im


def get_image_grid_numpy(
    input_data: torch.Tensor,
    target_data: torch.Tensor,
    outputs: torch.Tensor,
    classes: Sequence[str],
    batch_size: Optional[int] = None,
    fig_size: int = 3,
    clim: Optional[Sequence] = None,
    cmap: Optional[str] = None,
) -> np.ndarray:  # type: ignore
    """
    Create a grid of images for input, target, and output data using matplotlib and convert it to a numpy array.
    Args:
        input_data (torch.Tensor): Input data.
        target_data (torch.Tensor): Target data.
        outputs (torch.Tensor): Model outputs.
        classes (list): List of class labels.
        batch_size (int, optional): Number of images to display. Defaults to the length of the first axis of 'input_data'.
        fig_size (int, optional): Size of the figure. Defaults to 3.
        clim (tuple, optional): Color limits for the images. Defaults to be scaled by the image's intensity.
        cmap (str, optional): Colormap for the images. Defaults to None.

    Returns:
        fig (numpy.ndarray): Image data.
    """
    fig = get_image_grid(
        input_data=input_data,
        target_data=target_data,
        outputs=outputs,
        classes=classes,
        batch_size=batch_size,
        fig_size=fig_size,
        clim=clim,
        cmap=cmap,
    )
    return fig_to_image(fig)


def get_fig_dict(
    input_data: torch.Tensor,
    target_data: torch.Tensor,
    outputs: torch.Tensor,
    classes: Sequence[str],
    batch_size: Optional[int] = None,
    fig_size: int = 3,
    clim: Optional[Sequence] = None,
    colorbar: bool = True,
    colorbar_size: float = 0.1,
    gt_clim: Optional[Sequence] = (0, 1),
) -> dict:
    """
    Create a dictionary of figures for input, target, and output data.
    Args:
        input_data (torch.Tensor): Input data.
        target_data (torch.Tensor): Target data.
        outputs (torch.Tensor): Model outputs.
        classes (list): List of class labels.
        batch_size (int, optional): Number of images to display. Defaults to the length of the first axis of 'input_data'.
        fig_size (int, optional): Size of the figure. Defaults to 3.
        clim (tuple, optional): Color limits for the images. Defaults to be scaled by the image's intensity.
        colorbar (bool, optional): Whether to display a colorbar for the model outputs. Defaults to True.
        colorbar_size (float, optional): Size of the colorbar. Defaults to 0.2.
        gt_clim (tuple, optional): Color limits for the ground truth images. Defaults to (0, 1).

    Returns:
        image_dict (dict): Dictionary of figure objects.
    """
    if batch_size is None:
        batch_size = input_data.shape[0]
    image_dict = {}
    for c, label in enumerate(classes):
        if colorbar:
            grid_spec_kw = {"width_ratios": [1, 1, 1, 1, colorbar_size]}
        else:
            grid_spec_kw = {}
        fig, ax = plt.subplots(
            batch_size,
            4 + colorbar,
            figsize=(fig_size * (4 + colorbar * colorbar_size), fig_size * batch_size),
            gridspec_kw=grid_spec_kw,
        )
        if len(ax.shape) == 1:
            ax = ax[None, :]
        for b in range(batch_size):
            output = outputs[b][c].squeeze().cpu().detach().numpy()
            target = target_data[b][c].squeeze().cpu().detach().numpy()
            if len(output.shape) == 3:
                output_mid = output.shape[0] // 2
                output = output[output_mid]
                target = target[output_mid]
            ax[b, 2].imshow(target, clim=gt_clim)
            ax[b, 2].axis("off")
            ax[b, 2].set_title(f"GT {label}")
            im = ax[b, 3].imshow(output, clim=clim)
            ax[b, 3].axis("off")
            ax[b, 3].set_title(f"Pred. {label}")
            if colorbar:
                orientation = "vertical"
                location = "right"
                cbar = fig.colorbar(
                    im, orientation=orientation, location=location, cax=ax[b, 4]
                )
                ax[b, 4].set_title("Intensity")
            input_img = input_data[b][0].squeeze().cpu().detach().numpy()
            if len(input_img.shape) == 3:
                input_img = input_img[input_img.shape[0] // 2]
            x_pad, y_pad = (input_img.shape[0] - output.shape[0]) // 2, (
                input_img.shape[1] - output.shape[1]
            ) // 2
            if x_pad <= 0:
                x_slice = slice(0, input_img.shape[0])
            else:
                x_slice = slice(x_pad, -x_pad)
            if y_pad <= 0:
                y_slice = slice(0, input_img.shape[1])
            else:
                y_slice = slice(y_pad, -y_pad)
            ax[b, 1].imshow(input_img[x_slice, y_slice], cmap="gray", clim=clim)
            ax[b, 1].axis("off")
            ax[b, 1].set_title("Raw")
            ax[b, 0].imshow(input_img, cmap="gray", clim=clim)
            ax[b, 0].axis("off")
            ax[b, 0].set_title("Full FOV")
            w, h = output.shape[1], output.shape[0]
            rect = plt.Rectangle(  # type: ignore
                (x_pad, y_pad), w, h, edgecolor="r", facecolor="none"
            )
            ax[b, 0].add_patch(rect)
        fig.tight_layout()
        image_dict[label] = fig
    return image_dict


def get_image_dict(
    input_data: torch.Tensor,
    target_data: torch.Tensor,
    outputs: torch.Tensor,
    classes: Sequence[str],
    batch_size: Optional[int] = None,
    fig_size: int = 3,
    clim: Optional[Sequence] = None,
    colorbar: bool = True,
) -> dict:
    """
    Create a dictionary of images for input, target, and output data.
    Args:
        input_data (torch.Tensor): Input data.
        target_data (torch.Tensor): Target data.
        outputs (torch.Tensor): Model outputs.
        classes (list): List of class labels.
        batch_size (int, optional): Number of images to display. Defaults to the length of the first axis of 'input_data'.
        fig_size (int, optional): Size of the figure. Defaults to 3.
        clim (tuple, optional): Color limits for the images. Defaults to be scaled by the image's intensity.
        colorbar (bool, optional): Whether to display a colorbar for the model outputs. Defaults to True.

    Returns:
        image_dict (dict): Dictionary of image data.
    """
    # TODO: Get list of figs for the batches, instead of one fig per class
    fig_dict = get_fig_dict(
        input_data=input_data,
        target_data=target_data,
        outputs=outputs,
        classes=classes,
        batch_size=batch_size,
        fig_size=fig_size,
        clim=clim,
        colorbar=colorbar,
    )
    return {k: fig_to_image(v) for k, v in fig_dict.items()}
