import torch
from torch import nn

# 1. Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# 2. Dataset nonlinear
X = torch.linspace(-5, 5, 100).reshape(-1, 1).to(device)
y = X ** 2

# 3. Neural Network
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
).to(device)

# 4. Loss dan optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# 5. Training
for epoch in range(3000):

    predictions = model(X)
    loss = loss_fn(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 300 == 0:
        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {loss.item():.6f}"
        )

# Evaluation
with torch.no_grad():
    test = torch.tensor(
        [[-4.5], [-3.3], [-1.5], [1.5], [3.3], [4.5]],
        device=device
    )

    actual = test ** 2
    predicted = model(test)

    mae = torch.mean(torch.abs(actual - predicted))

    print("\n=== MODEL EVALUATION ===")

    for i in range(len(test)):
        print(
            f"X: {test[i].item():.1f} | "
            f"Actual: {actual[i].item():.2f} | "
            f"Predicted: {predicted[i].item():.4f}"
        )

    print("MAE:", mae.item())

# 6. Testing
test = torch.tensor(
    [[-4.0], [-2.0], [0.0], [2.0], [4.0]],
    device=device
)

with torch.no_grad():
    predictions = model(test)

    print("\n=== PREDICTION ===")

    for i in range(len(test)):
        print(
            f"X: {test[i].item():.1f} | "
            f"Actual: {test[i].item() ** 2:.1f} | "
            f"Predicted: {predictions[i].item():.4f}"
        )