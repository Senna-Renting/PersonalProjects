from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import SGD
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BIRD_AMOUNT = 4
FILENAME = "25dutchbirdmodel.pt"

# Dataset we will use for the model
class BirdDataset(Dataset):
    def __init__(self, filename):
        df = pd.read_pickle(filename)
        # Get unique columns
        labels = dict.fromkeys(df.columns).keys()
        # Convert unique columns to label name 
        get_label = lambda x: re.sub('\d+-', '', x)
        self.labels = list(map(get_label, labels))
        # Get unique labels
        unique_labels = dict.fromkeys(self.labels).keys()
        self.unique = len(unique_labels)
        # Match data with the label
        self.data = [df[label].to_numpy() for label in labels]
        # Make the one-hot encoder
        self.encoding = dict()
        vector = torch.zeros(self.unique)
        for i,label in enumerate(unique_labels):
            vec = vector.clone()
            vec[i] = 1
            self.encoding[label] = vec
        
    def decode(self, vector):
        for label,vec in self.encoding.items():
            if torch.equal(vec, vector):
                return label

    def summary(self):   
        sizes = [spec.shape[2] for spec, _ in self]
        print(f"max: {max(sizes)}")
        print(f"min: {min(sizes)}")
        print(f"avg: {sum(sizes)/len(sizes)}")
        print(f"total: {len(sizes)}")

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]).unsqueeze(0), self.encoding[self.labels[idx]]
    
# CNN model that needs input shape (1 x 128 x ...) and returns output shape (1 x bird_amount x ...)
class BirdModel(nn.Module):
    def __init__(self, bird_amount=BIRD_AMOUNT):
        super(BirdModel, self).__init__()
        CONV_AMOUNT = 32
        self.KERNEL_WIDTH = 3
        self.STRIDE = 1
        self.MAX_SEQ = (12000-self.KERNEL_WIDTH+1)//self.STRIDE
        self.bird_amount = bird_amount
        self.conv = nn.Sequential(
            nn.Conv2d(1,CONV_AMOUNT,kernel_size=(128,self.KERNEL_WIDTH), stride=self.STRIDE),
            nn.BatchNorm2d(CONV_AMOUNT),
            nn.ReLU(),
            nn.Dropout2d(p=0.7))
        self.max_to_out = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,self.MAX_SEQ)),
            nn.Flatten(),
            nn.Linear(CONV_AMOUNT, bird_amount),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Convolution
        x = self.conv(x)
        # Apply Zero-Padding here
        pad_size = self.MAX_SEQ - x.shape[3]
        pad = nn.ZeroPad2d((0,pad_size,0,0))
        x = pad(x)
        # Global max pooling
        x = self.max_to_out(x)
        return x
    
class ParamTracker():
    def __init__(self, momentum=np.nan, batch_size=np.nan, 
                 kernel_width=np.nan, stride=np.nan, dropout=np.nan):
        self.momentum = momentum
        self.batch_size = batch_size
        self.dropout = dropout
        self.kernel_width = kernel_width
        self.stride = stride
        self.test_losses = list()
        self.test_accuracies = list()
        self.lrs = list()
        self.epochs = list()
        self.losses = list()
        self.accuracies = list()

    def add(self, epoch=np.nan,loss=np.nan,accuracy=np.nan,test_loss=np.nan,test_acc=np.nan, lr=np.nan):
        self.epochs.append(epoch)
        self.lrs.append(lr)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.test_accuracies.append(test_acc)
        self.test_losses.append(test_loss)
    def set(self, test_accuracy=np.nan, test_loss=np.nan):
        self.test_accuracy = test_accuracy
        self.test_loss = test_loss

    def show(self):
        fig, axs = plt.subplots(3,3, constrained_layout=True) #[plt.subplot(331), plt.subplot(332), plt.subplot(333)]
        [[ax.xaxis.set_major_locator(MaxNLocator(5, integer=True)) for ax in row] for row in axs]
        handles = list()
        labels = list()
        # Loss plots
        axs[0][0].set_title("Model loss")
        axs[0][0].set_xlabel("Epoch")
        axs[0][0].set_ylabel("Loss")
        axs[0][0].plot(self.epochs, self.losses, label="train")
        axs[0][0].plot(self.epochs, self.test_losses, label="test")
        ax_handles, ax_labels = axs[0][0].get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
        # Accuracy plots
        axs[0][1].set_title("Model accuracy")
        axs[0][1].set_xlabel("Epoch")
        axs[0][1].set_ylabel("Accuracy")
        axs[0][1].plot(self.epochs, self.accuracies, label="train")
        axs[0][1].plot(self.epochs, self.test_accuracies, label="test")
        # Learning rate plot
        axs[0][2].set_title("Learning rate")
        axs[0][2].set_ylabel("Lr")
        axs[0][2].set_xlabel("Epoch")
        axs[0][2].plot(self.epochs, self.lrs)
        # Dashboard info
        [ax.axis('off') for ax in axs[1]]
        center_pos = [.5,.5]
        center_settings = {"verticalalignment":"center", "horizontalalignment":"center"}
        axs[1][0].text(*center_pos,"Batch size: {:.2f}".format(self.batch_size), **center_settings)
        axs[1][1].text(*center_pos,"Momentum: {:.2f}".format(self.momentum), **center_settings)
        axs[1][2].text(*center_pos,"Dropout: {:.2f}".format(self.dropout), **center_settings)
        [ax.axis('off') for ax in axs[2]]
        axs[2][0].text(*center_pos,"Kernel width: {}".format(self.kernel_width), **center_settings)
        axs[2][2].text(*center_pos,"Stride size: {}".format(self.stride), **center_settings)

        plt.tight_layout(w_pad=0.45)
        fig.legend(handles, labels, bbox_to_anchor=(0.45,0.15), loc="lower left", title="Legend")
        plt.show()


def train(model, dataset, epoch=5, lr=1e-2, batch_size=40, momentum=0.9, gamma=1):
    torch.manual_seed(42)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    tracker = ParamTracker(momentum=momentum, batch_size=batch_size, kernel_width=model.KERNEL_WIDTH, stride=model.STRIDE)
    train_data, test_data = random_split(dataset, [0.9, 0.1])
    test_loader = DataLoader(test_data, shuffle=False)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=pad_collate)
    print("Training results model:")
    print("-"*24)
    # Training loop
    test_best_acc = 0
    for j in range(epoch):
        running_loss = 0
        running_acc = 0
        model.train()
        # Train model
        for data in train_loader:
            features, labels = data
            # Perform training
            optimizer.zero_grad()
            predictions = model(features)
            running_acc += accuracy(predictions, labels)
            loss = loss_fn(predictions, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        train_acc = running_acc / len(train_data) * batch_size
        train_loss = running_loss / len(train_data) * batch_size
        # Test model
        model.eval()
        running_loss = 0
        running_acc = 0
        for features, labels in test_loader:
            predictions = model(features)
            loss = loss_fn(predictions, labels)
            running_acc += accuracy(predictions, labels)
            running_loss += loss.item()
            test_loss = running_loss/len(test_data)
            test_acc = running_acc/len(test_data)
        # Auto-Save the best performing version of model
        if test_acc > test_best_acc:
            test_best_acc = test_acc
            torch.save(model.state_dict(), FILENAME)
        # Track parameters and show in terminal
        tracker.add(epoch=j+1, loss=train_loss, lr=scheduler.get_last_lr(), accuracy=train_acc, test_loss=test_loss, test_acc=test_acc)
        print('Epoch {} loss: {:.4f}, acc: {:.4f}, val acc: {:.4f}'.format(j + 1, train_loss, train_acc, test_acc))
        scheduler.step()
    # Show and save results
    tracker.show()

# Reshapes tensors in batch to the same output size, giving all padding zero values
def pad_collate(batch):
    max_size = max([item[0].shape[2] for item in batch])
    labels = [item[1] for item in batch]
    data = list()
    for features, _ in batch:
        pad = max_size - features.shape[2]
        if pad > 0:
            padding = torch.zeros((features.shape[0], features.shape[1], pad))
            result = torch.cat([features, padding],2)
            data.append(result)
        else:
            data.append(features)

    return [torch.stack(data), torch.stack(labels)]

def strip_collate(batch):
    min_size = min([item[0].shape[2] for item in batch])
    labels = [item[1] for item in batch]
    data = list()
    for features, _ in batch:
        length = features.shape[2] - min_size
        strip = list(range(features.shape[2]-length))
        if len(strip) > 0:
            result = features[:,:,strip]
            data.append(result)
        else:
            data.append(features)

    return [torch.stack(data), torch.stack(labels)]

def accuracy(preds, labels, threshold=0.75):
    total = len(preds)
    sum = 0
    for i,pred in enumerate(preds):
        choice = (pred > threshold).type(torch.float)
        if torch.equal(choice, labels[i]) and torch.sum(choice).numpy() > 0:
            sum += 1
    return sum/total


if __name__ == "__main__":
    model = BirdModel(bird_amount=26)
    dataset = BirdDataset('25DutchBirdsDataset.pkl')
    dataset.summary()

    # Train blank model
    train(model, dataset, epoch=200, batch_size=40, lr=0.02, momentum=0.9, gamma=1)

    # Re-train previous model
    # model.load_state_dict(torch.load(FILENAME))
    # train(model, dataset, epoch=20, batch_size=20, lr=1e-2, momentum=0.9, gamma=0.95)

