from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = "hanoi"

X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []
X_val_list = []
y_val_list = []

for i in range(4, 11):
  data = torch.load(f"data_{i}q{env}.pt")
  X, y = data.tensors

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

  if i != 10:
    X_train_list.append(F.pad(X_train, (0, 10-i), "constant", 0))
    y_train_list.append(y_train)
    X_test_list.append(F.pad(X_test, (0, 10-i), "constant", 0))
    y_test_list.append(y_test)
    X_val_list.append(F.pad(X_val, (0, 10-i), "constant", 0))
    y_val_list.append(y_val)
  else:
    X_train_list.append(X_train)
    y_train_list.append(y_train)
    X_test_list.append(X_test)
    y_test_list.append(y_test)
    X_val_list.append(X_val)
    y_val_list.append(y_val)

y_train_list[0] = y_train_list[0].unsqueeze(1)
y_test_list[0] = y_test_list[0].unsqueeze(1)
y_val_list[0] = y_val_list[0].unsqueeze(1)

X_train = torch.cat(X_train_list, dim=0)
X_test = torch.cat(X_test_list, dim=0)
X_val = torch.cat(X_val_list, dim=0)

y_train = torch.cat(y_train_list, dim=0)
y_test = torch.cat(y_test_list, dim=0)
y_val = torch.cat(y_val_list, dim=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("X_val shape:", X_val.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("y_val shape:", y_val.shape)

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = Image.fromarray(self.data[index].numpy(), 'RGB')
        if self.transform:
            data = self.transform(data)
        label = self.labels[index]

        return data, label.float()

# Hyperparams
batch_size = 32
lr = 0.001
num_epochs = 100
count = 0
loss_val_et = 1000

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CustomDataset(X_train, y_train, transform=transform)
test_dataset = CustomDataset(X_test, y_test, transform=transform)
validation_dataset = CustomDataset(X_val, y_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

class DinoV2Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

        self.head = nn.Linear(384, 1)

    def forward(self, x):
        x = self.dino(x)
        return self.head(x)

dinoV2 = DinoV2Regression().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(dinoV2.parameters(), lr=lr)

for epoch in range(num_epochs):
    dinoV2.train()
    running_loss_train = 0.0
    for i, data in enumerate(tqdm(train_loader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = dinoV2(inputs)
        loss = criterion(outputs, labels)
        running_loss_train += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss_train = running_loss_train / len(train_loader)
    dinoV2.eval()
    with torch.no_grad():
        running_loss_validation = 0.0

        for i, data in enumerate(tqdm(validation_loader)):
            val_inputs, val_labels = data
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            val_outputs = dinoV2(val_inputs)
            
            val_loss = criterion(val_outputs, val_labels)
            running_loss_validation += val_loss.item()

        avg_loss_validation = running_loss_validation / len(validation_loader)

        if loss_val_et > avg_loss_validation:
          loss_val_et = avg_loss_validation
          torch.save(dinoV2.state_dict(), "model_dino.pt")
          count = 0
        else:
          count += 1

        print(f'Epoca [{epoch + 1}/{num_epochs}], '
              f'Perdida Entrenamiento: {avg_loss_train:.8f}, '
              f'Perdida Validación: {avg_loss_validation:.8f}, ')

        if count == 3:
          break

print("Entrenamiento completado")

loaded_dinoV2 = DinoV2Regression().to(device)
loaded_dinoV2.load_state_dict(torch.load("model_dino.pt"))
loaded_dinoV2.eval()

y_pred = []
y_true = []

with torch.no_grad():
  for test_inputs, test_labels in test_loader:
      test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

      test_outputs = loaded_dinoV2(test_inputs)

      y_true.extend(test_labels.cpu().detach().numpy())
      y_pred.extend(test_outputs.cpu().detach().numpy())

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

print(f"MAE: {mean_absolute_error(y_true, y_pred)}")
print(f"MSE: {mean_squared_error(y_true, y_pred)}")
print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False)}")
print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred)}")
print(f"R2: {r2_score(y_true, y_pred)}")
