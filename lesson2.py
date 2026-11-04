
import torch
from torch import nn

# 1. Siapkan dataset
X = torch.tensor(
    [[1.0], [2.0], [3.0], [4.0], [5.0]]
)

y = torch.tensor(
    [[3.0], [5.0], [7.0], [9.0], [11.0]]
)

# 2. Buat neural network
model = nn.Linear(1, 1)

# 3. Pilih loss function dan optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

# 4. Training loop
for epoch in range(1000):
    # Prediksi
    predictions = model(X)

    # Hitung kesalahan prediksi
    loss = loss_fn(predictions, y)

    # Reset gradient
    optimizer.zero_grad()

    # Hitung gradient
    loss.backward()

    # Perbarui parameter model
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.6f}"
        )

# 5. Uji model
model.eval()

with torch.no_grad():
    test = torch.tensor([[10.0]])
    prediction = model(test)

print("\nInput:", 10)
print("Prediksi:", prediction.item())
print("Bobot:", model.weight.item())
print("Bias:", model.bias.item())
