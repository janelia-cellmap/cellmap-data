# %%
import torch
from cellmap_data import CellMapDataset, CellMapDataLoader
from torchvision.models import resnet18

# %%
# Define the dataset files to use
dataset_dict = {
    "train": {"raw": "train_data.zarr/raw", "gt": "train_data.zarr/gt", "weight": 1.0},
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
