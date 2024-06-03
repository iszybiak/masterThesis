import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import random

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

csv_files = "client2_unbalanced_7_3.csv"

# Ustawianie ziarna losowości dla PyTorch
torch.manual_seed(0)

# Ustawianie ziarna losowości dla numpy
np.random.seed(0)
random.seed(0)

def load_datasets(csv_files):
    trainloaders = []
    valloaders = []
    testloaders = []

    # Wczytaj dane z pliku CSV
    data = pd.read_csv(csv_files)

    # Podziel dane na cechy (X) i etykiety (y)
    X = data.drop('diagnosis', axis=1).values
    y = data['diagnosis'].values

    # Podziel dane na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Podziel zbiór treningowy na treningowy i walidacyjny
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Konwertuj dane na tensory PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Utwórz obiekty DataLoader dla danych treningowych, walidacyjnych i testowych
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    trainloader = DataLoader(train_dataset, shuffle=True)
    valloader = DataLoader(val_dataset)
    testloader = DataLoader(test_dataset)

    trainloaders.append(trainloader)
    valloaders.append(valloader)
    testloaders.append(testloader)

    return trainloaders, valloaders, testloaders


trainloaders, valloaders, testloaders = load_datasets(csv_files)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(30, 120)  # 30 wejść (liczba cech), 120 wyjść w pierwszej warstwie liniowej
        self.fc2 = nn.Linear(120, 84)  # 120 wejść, 84 wyjścia w drugiej warstwie liniowej
        self.fc3 = nn.Linear(84, 2)  # 84 wejścia, 2 wyjścia w trzeciej warstwie liniowej

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Pierwsza warstwa liniowa z funkcją aktywacji ReLU
        x = F.relu(self.fc2(x))  # Druga warstwa liniowa z funkcją aktywacji ReLU
        x = self.fc3(x)  # Trzecia warstwa liniowa bez funkcji aktywacji (wyjście)
        return x


def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()  # Funkcja kosztu - Cross Entropy Loss
    optimizer = torch.optim.Adam(net.parameters())  # Optymalizator - Adam optimizer
    net.train()  # Ustawienie sieci w trybie trenowania

    for epoch in range(epochs):  # Pętla przez określoną liczbę epok
        for features, labels in trainloader:  # Pętla przez dane treningowe
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()  # Wyzerowanie gradientów z poprzednich iteracji
            outputs = net(features)  # Przód: obliczenie predykcji sieci
            loss = criterion(outputs, labels.long())  # Obliczenie funkcji kosztu
            loss.backward()  # Wsteczna propagacja: obliczenie gradientów
            optimizer.step()  # Aktualizacja parametrów modelu na podstawie gradientów


def test(net, testloader):
    """Evaluate the network on the test set."""
    criterion = torch.nn.CrossEntropyLoss()  # Funkcja kosztu - Cross Entropy Loss
    correct, total, loss = 0, 0, 0.0  # Inicjalizacja zmiennych dla metryk
    tp, tn, fp, fn = 0, 0, 0, 0  # Inicjalizacja zmiennych dla TP, TN, FP, FN
    net.eval()  # Ustawienie sieci w trybie oceny

    with torch.no_grad():  # Wyłączenie obliczeń gradientu podczas oceny
        for features, labels in testloader:  # Pętla przez dane testowe
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = net(features)  # Przód: obliczenie predykcji sieci
            labels = labels.long()
            loss += criterion(outputs, labels).item()  # Akumulacja straty
            total += labels.size(0)  # Liczenie całkowitej liczby etykiet
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()  # Obliczenie liczby poprawnych predykcji

            # Obliczenie TP, TN, FP, FN
            pred_labels = torch.argmax(outputs, dim=1)
            tp += ((pred_labels == 1) & (labels == 1)).sum().item()
            tn += ((pred_labels == 0) & (labels == 0)).sum().item()
            fp += ((pred_labels == 1) & (labels == 0)).sum().item()
            fn += ((pred_labels == 0) & (labels == 1)).sum().item()

    loss /= len(testloader.dataset)  # Obliczenie średniej straty
    accuracy = correct / total  # Obliczenie dokładności

    # Obliczenie czułości (sensitivity) i specyficzności (specificity)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return loss, accuracy, sensitivity, specificity  # Zwrócenie straty i dokładności


def print_metrics_table(loss, accuracy, sensitivity, specificity):
    df = pd.DataFrame({'Loss': [round(loss, 3)],
                       'Accuracy': [round(accuracy, 3)],
                       'Sensitivity': [round(sensitivity, 3)],
                       'Specificity': [round(specificity, 3)]})
    print(df)


def plot_learning_progress(losses, accuracies, sensitivities, specificities):
    num_rounds = len(losses)
    rounds = np.arange(1, num_rounds + 1)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rounds, losses, marker='o', color='b', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.grid(True)
    plt.legend()

    # Plot accuracies, sensitivities, and specificities
    plt.subplot(1, 2, 2)
    plt.plot(rounds, accuracies, marker='o', color='g', label='Accuracy')
    plt.plot(rounds, sensitivities, marker='o', color='r', label='Sensitivity')
    plt.plot(rounds, specificities, marker='o', color='m', label='Specificity')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Metrics over epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


trainloader = trainloaders[0]  # Assuming trainloaders is a list of train loaders, selecting the first one
testloader = testloaders[0]
valloader = valloaders[0]  # Assuming valloaders is a list of validation loaders, selecting the first one

net = Net().to(DEVICE)

# Initialize lists to store metrics
losses, accuracies, sensitivities, specificities = [], [], [], []

for epoch in range(30):  # Looping through 5 epochs
    train(net, trainloader, 1)  # Training the network for one epoch using the training loader
    loss, accuracy, sensitivity, specificity = test(net, valloader)  # Evaluating the network on the validation loader

    # Store metrics
    losses.append(loss)
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)

    print(f"Epoch {epoch + 1}:")
    print_metrics_table(loss, accuracy, sensitivity, specificity)

# Plot the learning progress
plot_learning_progress(losses, accuracies, sensitivities, specificities)

# Final test set performance
loss, accuracy, sensitivity, specificity = test(net, testloader)  # Evaluating the network on the test loader
print(
    f"\nFinal test set performance: sensitivity: {sensitivity * 100:.2f}%, specificity: {specificity * 100:.2f}%, accuracy: {accuracy * 100:.2f}%")  # Printing final test metrics
