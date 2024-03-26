# %%
import torch
from torchvision.models import resnet18

# %%
# Define the dataset files to use
dataset_dict = {
    # "train": {"raw": "train_data.zarr/raw", "gt": "train_data.zarr/gt", "weight": 1.0},
    "train": {"raw": "train_data.zarr/raw", "gt": "train_data.zarr/gt"},
    "val": {"raw": "val_data.zarr/raw", "gt": "val_data.zarr/gt"},
    "test": {"raw": "test_data.zarr/raw", "gt": "test_data.zarr/gt"},
}

# %%
# Create the dataset and dataloader
dataset = CellMapDataset(dataset_dict)
dataloader = CellMapDataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# %%
# Create the network
model = resnet18(num_classes=2)

# %%
# Define the loss function and optimizer
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
# Train the network
for epoch in range(10):
    for i, data in enumerate(dataloader):
        inputs, targets = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_value = loss(outputs, targets)
        loss_value.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Batch {i}, Loss {loss_value.item()}")
# %%
# Save the trained model
torch.save(model.state_dict(), "model.pth")

# =============================================================================
# %%
import zarrdataset as zds
import zarr
import matplotlib.pyplot as plt
import cellmap_data

filenames = ["/misc/public/dacapo_learnathon/jrc_hela-2.zarr"]

image_specs = zds.ImagesDatasetSpecs(
    filenames=filenames,
    data_group="recon-1/em/fibsem-uint8/s1",
    source_axes="ZYX",
)

# A list with a labeled image, for the single image in the dataset, is passed as `filenames` argument.
# filenames = ["/nrs/cellmap/bennettd/data/jrc_hela-2/jrc_hela-2.zarr"]
label_group = "recon-1/labels/groundtruth/crop155/mito/s0"
# label_group = "recon-1/labels/groundtruth/crop7/mito/s0"
# label_group = "recon-1/labels/groundtruth/crop6/mito/s0"
labels_specs = zds.LabelsDatasetSpecs(
    filenames=filenames,
    data_group=label_group,
    source_axes="ZYX",
)

# %%
patch_size = dict(Z=128, Y=128, X=128)
patch_sampler = zds.PatchSampler(patch_size=patch_size, min_area=0.1)

my_dataset = zds.ZarrDataset(
    # [image_specs, image_specs],
    [image_specs, labels_specs],
    patch_sampler=patch_sampler,
    # shuffle=True,
    # draw_same_chunk=True,
    return_positions=True,
)
# %%
fig, ax = plt.subplots(3, 6)
for i, (pos, sample, label) in enumerate(my_dataset):
    print(
        f"Sample {i}, patch size: {sample.shape}, label: {label.shape}, from position {pos}"
    )

    ax[i // 3, 2 * (i % 3)].imshow(sample[0], cmap="gray")
    ax[i // 3, 2 * (i % 3)].set_title(f"Image {i}")
    ax[i // 3, 2 * (i % 3)].axis("off")

    ax[i // 3, 2 * (i % 3) + 1].imshow(label[0])
    ax[i // 3, 2 * (i % 3) + 1].set_title(f"Label {i}")
    ax[i // 3, 2 * (i % 3) + 1].axis("off")

    # Obtain only 9 samples
    if i >= 8:
        break

plt.show()

# %%
from fibsem_tools import read, read_xarray
from pathlib import Path

filenames = ["/misc/public/dacapo_learnathon/jrc_hela-2.zarr"]
label_group = "recon-1/labels/groundtruth/crop155/mito"
raw_group = "recon-1/em/fibsem-uint8"

raw = read_xarray(Path(filenames[0], raw_group))
print(raw)
# %%

labels = read_xarray(Path(filenames[0], label_group))
print(labels)
# %%
