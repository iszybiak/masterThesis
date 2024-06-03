from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random

import flwr as fl
from flwr.common import Metrics

import pandas as pd
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cpu") # Try "cuda" to train on GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

CLASSES = (0,1)
NUM_CLIENTS = 3
BATCH_SIZE = 32 # bez ograniczenia wsadu

# Pliki CSV reprezentujące dane klientów
csv_files = ["client1.csv", "client2_unbalanced_8_2.csv", "client3.csv"]

# Ustawianie ziarna losowości dla PyTorch na CPU lub GPU
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Ustawianie ziarna losowości dla numpy
np.random.seed(0)

# Ustawianie ziarna losowości dla funkcji losujących w Pythonie
random.seed(0)
def plot_learning_progress(losses, accuracies, sensitivities, specificities):
    num_rounds = len(losses)
    rounds = np.arange(1, num_rounds + 1)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rounds, losses, marker='o', color='b', label='Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Loss over rounds')
    plt.grid(True)
    plt.legend()

    # Plot accuracies, sensitivities, and specificities
    plt.subplot(1, 2, 2)
    plt.plot(rounds, accuracies, marker='o', color='g', label='Accuracy')
    plt.plot(rounds, sensitivities, marker='o', color='r', label='Sensitivity')
    plt.plot(rounds, specificities, marker='o', color='m', label='Specificity')
    plt.xlabel('Round')
    plt.ylabel('Metric Value')
    plt.title('Metrics over rounds')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def load_datasets(csv_files):
    trainloaders = []
    valloaders = []
    testloaders = []

    for file in csv_files:
        # Wczytaj dane z pliku CSV
        data = pd.read_csv(file)

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
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(30, 120)  # Pierwsza warstwa liniowa: wejście o wymiarze 30, wyjście 120
        self.fc2 = nn.Linear(120, 84)  # Druga warstwa liniowa: wejście 120, wyjście 84
        self.fc3 = nn.Linear(84, 2)  # Trzecia warstwa liniowa: wejście 84, wyjście 2

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Pierwsza warstwa liniowa + funkcja aktywacji ReLU
        x = F.relu(self.fc2(x))  # Druga warstwa liniowa + funkcja aktywacji ReLU
        x = self.fc3(x)  # Trzecia warstwa liniowa (wyjście)
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
            total += labels.size(0)    # Liczenie całkowitej liczby etykiet
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
    df = pd.DataFrame({ 'Loss': [round(loss, 3)],
                        'Accuracy': [round(accuracy, 3)],
                        'Sensitivity': [round(sensitivity, 3)],
                        'Specificity': [round(specificity, 3)]})
    print(df)

net = Net().to(DEVICE)

for i in range(3):
    print(f"Client {i+1}:")
    trainloader = trainloaders[i]
    valloader = valloaders[i]
    testloader = testloaders[i]

    # Trening i walidacja przez 40 epok
    for epoch in range(30):
        train(net, trainloader, 1)  # Trening sieci przez jedną epokę
        loss, accuracy, sensitivity, specificity = test(net, valloader)  # Walidacja sieci
        print(f"Epoch {epoch + 1}:")
        print_metrics_table(loss, accuracy, sensitivity, specificity)

    # Testowanie na testloaderze
    loss, accuracy, sensitivity, specificity = test(net, testloader)
    print(f"\nFinal test set performance for dataset {i+1}: sensitivity: {sensitivity * 100:.2f}%, specificity: {specificity * 100:.2f}%, accuracy: {accuracy * 100:.2f}%\n")

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy, sensitivity, specificity = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {
            "accuracy": float(accuracy),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
        }

def client_fn(cid: str) -> FlowerClient:
        net = Net().to(DEVICE)
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        return FlowerClient(net, trainloader, valloader)


# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=1.0,
#     fraction_evaluate=0.5,
#     min_fit_clients=3,
#     min_evaluate_clients=2,
#     min_available_clients=3,
# )

client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

# simulation_result = fl.simulation.start_simulation(
#     client_fn=client_fn,
#     num_clients=NUM_CLIENTS,
#     config=fl.server.ServerConfig(num_rounds=5),
#     strategy=strategy,
#     client_resources=client_resources,
# )

def weighted_average(metrics: List[Tuple[int,Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    sensitivities = [num_examples * m["sensitivity"] for num_examples, m in metrics]
    specificities = [num_examples * m["specificity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "accuracy": sum(accuracies) / sum(examples),
        "sensitivity": sum(sensitivities) / sum(examples),
        "specificity": sum(specificities) / sum(examples),
    }

def median_aggregation(metrics: List[Tuple[int,Metrics]]) -> Metrics:
    accuracies = [m["accuracy"] for _, m in metrics]
    sensitivities = [m["sensitivity"] for _, m in metrics]
    specificities = [m["specificity"] for _, m in metrics]

    median_accuracy = np.median(accuracies)
    median_sensitivity = np.median(sensitivities)
    median_specificity = np.median(specificities)

    return {
        "accuracy": median_accuracy,
        "sensitivity": median_sensitivity,
        "specificity": median_specificity,
    }

def weighted_median_aggregation(metrics: List[Tuple[int,Metrics]]) -> Metrics:
    accuracies = []
    sensitivities = []
    specificities = []
    weights = []

    for num_examples, m in metrics:
        accuracies.extend([m["accuracy"]] * num_examples)
        sensitivities.extend([m["sensitivity"]] * num_examples)
        specificities.extend([m["specificity"]] * num_examples)
        weights.extend([num_examples] * num_examples)

    weighted_median_accuracy = np.median(accuracies)
    weighted_median_sensitivity = np.median(sensitivities)
    weighted_median_specificity = np.median(specificities)

    return {
        "accuracy": weighted_median_accuracy,
        "sensitivity": weighted_median_sensitivity,
        "specificity": weighted_median_specificity,
    }

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average,
)

simulation_result=fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=30),
    strategy=strategy,
    client_resources=client_resources
)

result = simulation_result.metrics_distributed
losses = [val[1] for val in simulation_result.losses_distributed]
accuracies = [val[1] for val in result['accuracy']]
sensitivities = [val[1] for val in result['sensitivity']]
specificities = [val[1] for val in result['specificity']]


print("Losses", losses)
print("Accuracies:", accuracies)
print("Sensitivities:", sensitivities)
print("Specificities:", specificities)


plot_learning_progress(losses, accuracies, sensitivities, specificities)

avg_sensitivity = sum(sensitivities) / len(sensitivities)
avg_specificity = sum(specificities) / len(specificities)
avg_accuracy = sum(accuracies) / len(accuracies)

print(f"\nFinal test set performance : sensitivity: {avg_sensitivity * 100:.2f}%, specificity: {avg_specificity * 100:.2f}%, accuracy: {avg_accuracy * 100:.2f}%\n")
