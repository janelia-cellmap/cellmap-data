import matplotlib.pyplot as plt


def get_image_grid(
    input_data,
    target_data,
    outputs,
    classes,
    batch_size=None,
    fig_size=3,
    clim=None,
    cmap=None,
):
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
        rect = plt.Rectangle((x_pad, y_pad), w, h, edgecolor="r", facecolor="none")
        ax[b, 0].add_patch(rect)
    fig.tight_layout()
    return fig


def get_image_dict(
    input_data,
    target_data,
    outputs,
    classes,
    batch_size=None,
    fig_size=3,
    clim=None,
    colorbar=True,
):
    if batch_size is None:
        batch_size = input_data.shape[0]
    image_dict = {}
    for c, label in enumerate(classes):
        fig, ax = plt.subplots(
            batch_size, 4, figsize=(fig_size * 4, fig_size * batch_size)
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
            ax[b, 2].imshow(target, clim=clim)
            ax[b, 2].axis("off")
            ax[b, 2].set_title(f"GT {label}")
            ax[b, 3].imshow(output, clim=clim)
            ax[b, 3].axis("off")
            ax[b, 3].set_title(f"Pred. {label}")
            if colorbar and clim is None:
                if batch_size == 1:
                    orientation = "horizontal"
                    location = "bottom"
                else:
                    orientation = "vertical"
                    location = "right"
                ax[b, 3].colorbar(orientation=orientation, location=location)
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
            rect = plt.Rectangle((x_pad, y_pad), w, h, edgecolor="r", facecolor="none")
            ax[b, 0].add_patch(rect)
        fig.tight_layout()
        image_dict[label] = fig
    return image_dict
