# Filename: main.py
# Author: Liam Laidlaw & Brandon Rose
# Updated: 12-06-2024
# Description: The main driver program for the classifier

import random
import time
import torch

import random
import time
import torch
import torchaudio
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from network import Network
from rnn import RNN  # Import RNN class


def parse_wave_data(path: str):
    """
    Parses all the audio files in the genres_original directory.
    """
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    data = []

    for genre in genres:
        for i in range(100):
            if i < 10:
                filename = f"{genre}.0000{i}"
            else:
                filename = f"{genre}.000{i}"
            # Append tuple with tensor and genre label
            # data.append((torchaudio.load(f"{path}/{genre}/{filename}.wav"), genre))
    return data


def parse_csv_data(path: str, genre_to_value_map: dict):
    """
    Parses all the pre-extracted features into tensors.
    """
    data = []
    with open(path, "r") as file:
        file.readline()  # Skip the column labels
        for line in file:
            line = line.strip().split(",")
            filename = line[0]
            features = list(map(float, line[1:-1]))
            label = genre_to_value_map[line[-1]]
            # Append tuple to the data list
            data.append((torch.tensor(features), label, filename))
    return data


def rnn_data_prep(data: list[tuple[torch.Tensor, int]]) -> list[tuple[torch.Tensor, int]]:
    #converts data into sequence format -> batch, sequence length, feature size
    #ltsm requires input tensors to be 3D
    processed_data = []
    for features, label, *_ in data:
        seq_features = features.unsqueeze(0) if features.ndim == 1 else features
        processed_data.append((seq_features, label))
    return processed_data



def train_network(net: torch.nn.Module, data: list[tuple[torch.Tensor, int]], epochs: int, batch_size: int):
    #training function modified to handle different types of nns
    data = [(features, label) for features, label, *_ in data]
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    net.train()
    for epoch in range(epochs):
        global_loss = 0
        correct = 0
        for batch in loader:
            input, target = batch
            pred = net(input)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_loss += loss.item()
            pred = torch.argmax(pred, dim=1)
            correct += torch.sum(target == pred).item()
            scheduler.step()
        print(f"epoch: {epoch} -> loss: {round(global_loss, 6)} -> acc: {correct} of {len(data)} -> {round(correct / len(data) * 100, 3)}%")


def accuracy(net: torch.nn.Module, data: list[tuple[torch.Tensor, int]]) -> float:
    """
    Calculates the accuracy of a neural network on a given dataset.
    """
    data = [(features, label) for features, label, *_ in data]  # Strip filenames or extra elements
    loader = DataLoader(data, batch_size=32)
    correct = 0
    for batch in loader:
        inp, exp = batch
        pred = net(inp)
        pred = torch.argmax(pred, dim=1)
        correct += torch.sum(exp == pred).item()
    return correct / len(data)


def main():
    value_to_genre_map = {
        0: "blues",
        1: "classical",
        2: "country",
        3: "disco",
        4: "hiphop",
        5: "jazz",
        6: "metal",
        7: "pop",
        8: "reggae",
        9: "rock"
    }

    genre_to_value_map = {v: k for k, v in value_to_genre_map.items()}

    print("Loading and preprocessing data")
    three_sec_csv_data = parse_csv_data("./GTZAN/features_3_sec.csv", genre_to_value_map)
    thirty_sec_csv_fata = parse_csv_data("./GTZAN/features_30_sec.csv", genre_to_value_map)

    # Shuffle data
    random.seed(42)
    random.shuffle(three_sec_csv_data)
    random.seed(time.time())

    # Split data
    train_data = three_sec_csv_data[:int(len(three_sec_csv_data) * (2 / 3))]
    test_data = three_sec_csv_data[int(len(three_sec_csv_data) * (2 / 3)):]

    train_data_30 = three_sec_csv_data[:int(len(three_sec_csv_data) * (2 / 3))]
    test_data_30 = three_sec_csv_data[int(len(three_sec_csv_data) * (2 / 3)):]

    # Preprocess data for RNN
    train_data_rnn = rnn_data_prep(train_data)
    test_data_rnn = rnn_data_prep(test_data)

    train_data_rnn_30 = rnn_data_prep(train_data_30)
    test_data_rnn_30 = rnn_data_prep(test_data_30)

    # Define model parameters
    input_size = 58  # Adjust based on feature size
    hidden_size = 64
    output_size = len(value_to_genre_map)
    num_layers = 2
    epochs = 1000
    batch_size = 32

    # Define models
    print("Initializing models")
    linear_net = Network(layers=10, hidden_size=32, in_size=input_size, out_size=output_size)
    rnn_net = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
    rnn_net_30 = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)

    ## Train linear network
    #print("Training linear network...")
    #train_network(linear_net, train_data, epochs=epochs, batch_size=batch_size)

    # Train RNN network
    print("Training RNN network 3 seconds...")
    train_network(rnn_net, train_data_rnn, epochs=epochs, batch_size=batch_size)

    print("Training RNN network 30 seconds...")
    train_network(rnn_net_30, train_data_rnn_30, epochs=epochs, batch_size=batch_size)

    # Test models
    print("Evaluating models")
    #linear_accuracy = accuracy(linear_net, test_data)
    rnn_accuracy = accuracy(rnn_net, test_data_rnn)
    rnn_accuracy_30 = accuracy(rnn_net_30, test_data_rnn_30)

    # Display results
    #print(f"Linear Network accuracy: {linear_accuracy}")
    print(f"RNN accuracy 3secs: {rnn_accuracy}")
    print(f"RNN accuracy 30secs: {rnn_accuracy_30}")


if __name__ == '__main__':
    main()
