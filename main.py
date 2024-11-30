# Filename: main.py
# Author: Liam Laidlaw
# Created: 11-25-2024
# Description: The main driver program for the classifier

import random
import time
import torch

#import torchaudio
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from network import Network


def parse_wave_data(path: str):
    """
    purpose: parses all the audio files in the genres_original directory
    param path: the path to the .wav file that is being parsed
    returns: list of tuples -> (tensor, label)
    note: This function requires all of the default filenames and directories for the GTZAN dataset
    """
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    data = []

    for genre in genres: 
        for i in range(100):
            if i < 10:
                filename = f"{genre}.0000{i}"
            else:
                filename = f"{genre}.000{i}"
            # append tuple with tensor and genre label
            #data.append((torchaudio.load(f"{path}/{genre}/{filename}.wav"), genre))
    return data


def parse_csv_data(path: str, genre_to_value_map: dict):
    """
    purpose: parses all the pre extracted features into tensors 
    param path: the path to the .csv file that is being parsed
    returns: list of tuples -> (tensor, label, filename)
    note: This function requires all of the default filenames and directories for the GTZAN dataset
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
    purpose: trains the neural network 
    param net: the neural network object that is being trained
    param data: a list of tuples of torch tensors and labels
    param epochs: the number of iterations through the training data the network will complete
    param batch_size: the number of items in a data batch 
    returns: the argued torch neural network module
    """

    # ignore file names in each tuple
    data_no_filenames = [(pair[0], pair[1]) for pair in data]

    loader = DataLoader(data_no_filenames, batch_size=batch_size, shuffle=True)
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
            # perform back propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() # zero all the weight gradients in preparation for the next batch
            
            # Compute global loss
            global_loss += loss.item()
            # Compute epoch accuracy
            pred = torch.argmax(pred, dim=1)
            correct += torch.sum(target == pred).item()
            scheduler.step()
        print(f"epoch: {epoch} -> loss: {round(global_loss, 6)} -> acc: {correct} of {len(data_no_filenames)} -> {round(correct/len(data_no_filenames)*100, 3)}%")
    return net

def accuracy(net: torch.nn.Module, data: list[tuple[torch.Tensor, int]]) -> float:
    """
    purpose: calculates the accuracy of a neural network on a given dataset
    param net: the network that is being tested
    param data: the dataset used for testing
    returns: a floating point value as the proportion of correct predictions that the network made
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
    # output of the nn will be mapped against this dictionary
    value_to_genre_map = {
        0 : "blues", 
        1 : "classical", 
        2 : "country", 
        3 : "disco", 
        4 : "hiphop", 
        5 : "jazz", 
        6 : "metal", 
        7 : "pop", 
        8 : "reggae", 
        9 : "rock"
    }

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

    num_linear_layers_csv = 10
    linear_width_csv = 20
    in_size_csv = 58 # number of features present in each csv file
    out_size_csv = len(value_to_genre_map) # number of categories to map to
    epochs = 1000
    batch_size = 15

    # generate networks, one trained on 3 second features one trained on 30 second features
    linear_net_3_sec_csv = Network(num_linear_layers_csv, linear_width_csv, in_size_csv, out_size_csv)
    linear_net_30_sec_csv = Network(num_linear_layers_csv, linear_width_csv, in_size_csv, out_size_csv)
    
    # traing networks
    print("Traning 3 sec network...")
    train_network(linear_net_3_sec_csv, three_sec_csv_training, epochs=epochs, batch_size=batch_size)

    print("Training 30 sec network...")
    train_network(linear_net_30_sec_csv, thirty_sec_csv_training, epochs=epochs, batch_size=batch_size)

    # get the accuracy for each network on their test data
    three_sec_accuracy = accuracy(linear_net_3_sec_csv, three_sec_csv_test)
    thirty_sec_accuracy = accuracy(linear_net_30_sec_csv, thirty_sec_csv_test)
    print()
    print(f"3 second accuracy: {three_sec_accuracy}")
    print(f"30 second accuracy: {thirty_sec_accuracy}")


if __name__ == '__main__':
    main()