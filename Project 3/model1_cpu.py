import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.sampler
import torch.nn.functional as F


class MNIST:
    image_shape = (1, 28, 28)
    num_classes = 10

    @classmethod
    def _get_images(cls, data):
        x = data
        x = x.reshape((-1, *cls.image_shape))
        x = x.astype(np.float32)
        x /= 255
        return x

    def __init__(self, input_file, train=True):

        trainSet = np.load(input_file)
        np.random.shuffle(trainSet)

        if train:
            self.labels = trainSet[:, 1].astype(int)
            self.images = self._get_images(trainSet[:, 2:])
        else:
            self.images = self._get_images(trainSet[:, 1:])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if hasattr(self, 'labels'):
            return self.images[idx, :], self.labels[idx]
        else:
            return self.images[idx, :]


class DigitRecognizerCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5), stride=1, padding=2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ModelCNN(nn.Module):

    def __init__(self):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (5, 5))
        self.conv2 = nn.Conv2d(16, 32, (5, 5))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 512)
        h3 = F.relu(self.fc1(h2))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = self.fc2(h3)
        return h4


def train(model, loss_fn, optimizer, train_batches, cv_batches, device, num_epochs=30, status_every=5):
    losses_scores = []

    for epoch in range(num_epochs):

        epoch_losses = []

        for images, labels in train_batches:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            
            labels = labels.long()

            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        epoch_scores = []

        for images, labels in cv_batches:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            _, labels_pred = logits.max(dim=1)
            #score = (labels_pred == labels).float().mean()
            
            #print(labels_pred)
            #print(labels)
            
            score = (labels_pred == labels.long()).float().mean()
            epoch_scores.append(score.item())

        losses_scores.append({'epoch': epoch,
                              'loss': epoch_losses,
                              'score': epoch_scores})

        if epoch % status_every == 0 or epoch == num_epochs - 1:
            print(f'epoch={epoch:g}, '
                  f'loss={np.mean(epoch_losses):g}, '
                  f'val_acc={np.mean(epoch_scores):g}, '
                  f'ac_std={np.std(epoch_scores):g}')

    return losses_scores


def train_model(train_batches, cv_batches, device, num_epochs=30, learning_rate=1e-4, weight_decay=1e-3):
    model = ModelCNN()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)


    losses_scores = train(model,
                          loss_fn,
                          optimizer,
                          train_batches,
                          cv_batches,
                          device,
                          num_epochs=num_epochs)

    return losses_scores, model


def get_train_cv_indices(num_examples, train_fraction, random_seed=42):
    np.random.seed(random_seed)
    indices = np.random.permutation(num_examples)
    train_examples = int(train_fraction * num_examples)
    train_indices = indices[:train_examples]
    cv_indices = indices[train_examples:]
    return train_indices, cv_indices


def get_data_loader(data, indices, batch_size):
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    return torch.utils.data.DataLoader(
        data, sampler=sampler, batch_size=batch_size)



device = torch.device('cpu')

data = MNIST( './trainingSet.npy')

train_indices, cv_indices = get_train_cv_indices(len(data), train_fraction=0.95, random_seed=42)
train_batches = get_data_loader(data, train_indices, batch_size=256)
cv_batches = get_data_loader(data, cv_indices, batch_size=256)

print(f'Training on {len(train_indices)} examples. Cross-validating with {len(cv_indices)} examples')

losses_scores, model = train_model(train_batches, cv_batches, device, num_epochs=120, learning_rate=1e-4,
                                   weight_decay=0.003)
plt.figure(figsize=(11, 7))
for metric in ['loss', 'score']:
    (pd.concat({d['epoch']: pd.Series(d[metric], name=metric)
                for d in losses_scores},
               names=['epoch'])
     .groupby('epoch').mean()
     .plot(label=metric))

plt.axhline(0, ls='--')
plt.axhline(1, ls='--')
plt.savefig('Train.png')

