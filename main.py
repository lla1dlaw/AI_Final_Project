# Filename: main.py
# Author: Liam Laidlaw
# Created: 11-25-2024
# Description: The main driver program for the linear feed forward network music classifier

import random
import time
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from network import Network


def parse_csv_data(path: str, genre_to_value_map: dict):
    """
    Parses all the pre extracted features into tensors 
    :param path: the path to the .csv file that is being parsed
    :returns: list of tuples -> (tensor, label, filename)
    :note: This function requires all of the default filenames and directories for the GTZAN dataset
    """
    data = []

    with open(path, "r") as file:
        file.readline() # scrap the collumn labels
        for ln in file: 
            line = file.readline()
            line = line.strip().split(",")
            # build tuple
            filename = line[0]
            features = list(map(float, line[1:-1]))
            label = genre_to_value_map[line[-1]]
            # append tuple to the data list
            data.append((torch.tensor(features), label, filename))
        file.close()
    
    return data


def train_network(net: torch.nn.Module, data: list[tuple[torch.Tensor, int, str]], epochs: int = 100, batch_size: int = 32) -> torch.nn.Module:
    """
    Trains the neural network 
    :param net: the neural network object that is being trained
    :param data: a list of tuples of torch tensors and labels
    :param epochs: the number of iterations through the training data the network will complete
    :param batch_size: the number of items in a data batch 
    :returns: the argued torch neural network module
    """

    # ignore file names in each tuple
    data_no_filenames = [(pair[0], pair[1]) for pair in data]

    loader = DataLoader(data_no_filenames, batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    net.train()
    for epoch in range(epochs):
        global_loss = 0
        correct = 0
        for batch in loader:
            input, target = batch
            pred = net(input)
            loss = loss_fn(pred, target)
            # perform back propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() # zero all the weight gradients in preparation for the next batch
            
            # Compute global loss
            global_loss += loss.item()
            # Compute epoch accuracy
            pred = torch.argmax(pred, dim=1)
            correct += torch.sum(target == pred).item()
        print(f"epoch: {epoch} -> loss: {round(global_loss, 6)} -> acc: {correct} of {len(data_no_filenames)} -> {round(correct/len(data_no_filenames)*100, 3)}%")
    return net


def accuracy(net: torch.nn.Module, data: list[tuple[torch.Tensor, int]]) -> float:
    """
    Calculates the accuracy of a neural network on a given dataset
    :param net: the network that is being tested
    :param data: the dataset used for testing
    :returns: a floating point value as the proportion of correct predictions that the network made
    """

    # ignore file names in each tuple
    data_no_filenames = [(pair[0], pair[1]) for pair in data]

    loader = DataLoader(data_no_filenames, batch_size=32)
    correct = 0
    for batch in loader:
        inp, exp = batch
        pred = net(inp)
        # Compute accuracy
        pred = torch.argmax(pred, dim=1)
        correct += torch.sum(exp == pred).item()
    return correct / len(data)


def main():
    # used for mapping labels to values in torch tensors for accuracy calcualtion
    genre_to_value_map = {
        "blues" : 0, 
        "classical" : 1, 
        "country" : 2, 
        "disco" : 3, 
        "hiphop" : 4, 
        "jazz" : 5, 
        "metal" : 6, 
        "pop" : 7, 
        "reggae" : 8, 
        "rock" : 9
    }

    print("Creating linear network")
    # gather data
    three_sec_csv_data = parse_csv_data("./GTZAN/features_3_sec.csv", genre_to_value_map)
    thirty_sec_csv_data = parse_csv_data("./GTZAN/features_30_sec.csv", genre_to_value_map)
    
    # shuffle data
    random.seed(42)
    random.shuffle(three_sec_csv_data)
    random.shuffle(thirty_sec_csv_data)
    random.seed(time.time())

    # segment traing and test data (2/3 for training, 1/3 for testing)
    three_sec_csv_training = three_sec_csv_data[:int(len(three_sec_csv_data)*(2/3))]
    three_sec_csv_test = three_sec_csv_data[int(len(three_sec_csv_data)*(2/3)):]

    thirty_sec_csv_training = thirty_sec_csv_data[:int(len(thirty_sec_csv_data)*(2/3))]
    thirty_sec_csv_test = thirty_sec_csv_data[int(len(thirty_sec_csv_data)*(2/3)):]

    num_linear_layers_csv = 8
    linear_width_csv = 5
    in_size_csv = 58 # number of features present in each csv file
    out_size_csv = len(genre_to_value_map) # number of categories to map to
    epochs = 1000
    batch_size = 15

    # generate networks, one trained on 3 second features one trained on 30 second features
    linear_nets_3_sec_csv = [Network(num_linear_layers_csv*i, linear_width_csv*i, in_size_csv, out_size_csv) for i in range(1, 4)]
    linear_nets_30_sec_csv = [Network(num_linear_layers_csv*i, linear_width_csv*i, in_size_csv, out_size_csv) for i in range(1, 4)]

    # train linear networks
    for i, net in enumerate(linear_nets_3_sec_csv):
        print(f"Training 3 sec network {i+1}")
        train_network(net, three_sec_csv_training, epochs=epochs, batch_size=batch_size)
    
    for i, net in enumerate(linear_nets_30_sec_csv):
        print(F"Training 30 sec network {i+1}")
        train_network(net, thirty_sec_csv_training, epochs=epochs, batch_size=batch_size)

    # get the accuracy for each network on their test data
    # dummy networks
    best_3_sec_network = Network(1, 1, 1, 1)
    best_3_sec_network.accuracy = 0
    best_30_sec_network = Network(1, 1, 1, 1)
    best_30_sec_network.accuracy = 0

    # determine most accurate network
    for net in linear_nets_3_sec_csv:
        net.accuracy = accuracy(net, three_sec_csv_test)
        if net.accuracy > best_3_sec_network.accuracy:
            best_3_sec_network = net
    
    for net in linear_nets_30_sec_csv:
        net.accuracy = accuracy(net, thirty_sec_csv_test)
        if net.accuracy > best_30_sec_network.accuracy:
            best_30_sec_network = net

    # print save the model with the best accuracy
    print(f"\nBest 3 second accuracy: {best_3_sec_network.accuracy} -> Num Hidden Layers: {best_3_sec_network.num_layers} -> Hidden Layer Width: {best_3_sec_network.hidden_size}")
    print(f"Best 30 second accuracy: {best_30_sec_network.accuracy} -> Num Hidden Layers: {best_30_sec_network.num_layers} -> Hidden Layer Width: {best_30_sec_network.hidden_size}")

    # save the best networks
    torch.save(best_3_sec_network.state_dict(), "Best_Performing_Networks/best_3_second_net.pth")
    torch.save(best_30_sec_network.state_dict(), "Best_Performing_Networks/best_30_second_net.pth")


if __name__ == '__main__':
    main()
