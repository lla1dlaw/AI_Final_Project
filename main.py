# Filename: main.py
# Author: Liam Laidlaw
# Created: 11-25-2024
# Description: The main driver program for the classifier

import torch
#import torchaudio
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
            data.append((torchaudio.load(f"{path}/{genre}/{filename}.wav"), genre))
    return data


def parse_csv_data(path: str):
    data = []

    with open(path, "r") as file:
        file.readline()
        for line in file: 
            line = line.strip().split(",")
            features = map(float, line[:-1])
            label = line[-1]
            data.append((torch.tensor(features), label))

def main():
    num_linear_layers = 0
    linear_width = 0
    in_size = 0
    out_size = 0 

    genre_map = {
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

    print("Creating linear network")
    linear_net = Network(num_linear_layers, linear_width, in_size, out_size)

    # waveform_data = parse_wave_data("./GTZAN/genres_original")
    # print(waveform_data)
    csv_data = parse_csv_data("GTZAN/")
    


if __name__ == '__main__':
    main()