# Filename: cnnmain.py
# Author: Mario
# Created: 12-6-2024
# Description: The main driver program for the cnn


import torch


from cnn import CNN
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def evaluate_model(model, test):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test:
            outputs = model(inputs)
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")


def train_model(model, trainLoader, criterion, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        runningLoss = 0.0
        correct = 0
        total = 0
        oldaccuracy = 0
        for inputs, labels in trainLoader:

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epochLoss = runningLoss / len(trainLoader)
        accuracy = correct / total * 100
        print(f"Epoch: {epoch + 1}/{epochs}, Loss: {epochLoss:.4f}, Accuracy: {accuracy:.2f}%")
        #to not overfit
        # if accuracy >= 95 and oldaccuracy >= accuracy:
        #     break
        # oldaccuracy = accuracy




def main():
    transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load the train and test data
    imageTrain = ImageFolder("images_train", transform=transform)
    imageTest = ImageFolder("images_test", transform=transform)
    trainLoader = torch.utils.data.DataLoader(imageTrain, batch_size=32,shuffle=True, num_workers=2)
    testLoader = torch.utils.data.DataLoader(imageTest, batch_size=32,shuffle=False, num_workers=2)
    classes = imageTrain.classes
    numGenres= len(classes)

    cnnimageModel = CNN(numGenres)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnnimageModel.parameters(), lr=0.001)


    #training
    train_model(cnnimageModel, trainLoader, criterion, optimizer,25)


    #eval
    evaluate_model(cnnimageModel, testLoader)

    torch.save(cnnimageModel.state_dict(),"MusicModel1")


if __name__ == '__main__':
    main()
