
import torch
from torch import nn

# 1. Pilih device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# 2. Dataset
X = torch.tensor(
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    device=device
)

y = torch.tensor(
    [[8.0], [11.0], [14.0], [17.0], [20.0]],
    device=device
)
# 3. Neural Network
model = nn.Linear(1, 1).to(device)

# 4. Loss dan optimizer
loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# 5. Training
for epoch in range(1000):

    predictions = model(X)

    loss = loss_fn(predictions, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {loss.item():.6f}"
        )

with torch.no_grad():
    predictions = model(X)

    print("\n=== HASIL PREDIKSI ===")

    for i in range(len(X)):
        print(
            f"X: {X[i].item():.1f} | "
            f"Actual: {y[i].item():.1f} | "
            f"Predicted: {predictions[i].item():.4f}"
        )

# 6. Testing
with torch.no_grad():
    test = torch.tensor([[10.0]], device=device)

    prediction = model(test)

    print("Prediction:", prediction.item())
    print("Weight:", model.weight.item())
    print("Bias:", model.bias.item())
