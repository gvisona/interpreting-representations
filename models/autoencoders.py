import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from copy import deepcopy


def chunker(data_array, batch_size=32):
    for i in range(0, len(data_array), batch_size):
        yield data_array[i:i+batch_size]

class Autoencoder(nn.Module):
    def __init__(self, encoder_layers, decoder_layers):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(torch.Tensor(x))
    
    def decode(self, x):
        return self.decoder(torch.Tensor(x))

    def layers_list(self):
        return "Encoder: " + str(self.encoder) + "\nDencoder: " + str(self.decoder)

def train_autoencoder(model, data, params):
    if not os.path.exists("trained_models"):
        os.mkdir("trained_models")
    folder_name = os.path.join("trained_models", "AE", time.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(folder_name)


    lr = params.get("lr", 1e-3)
    params["lr"] = lr
    wd = params.get("wd", 1e-4)
    params["wd"] = wd
    epochs = params.get("epochs", 20)
    params["epochs"] = epochs
    patience = params.get("patience", 3)
    params["patience"] = patience
    valid_split = params.get("valid_split", 0.1)
    params["valid_split"] = valid_split
    batch_size = params.get("batch_size", 64)
    params["batch_size"] = batch_size

    with open(os.path.join(folder_name, "params.json"), "w") as f:
        json.dump(params, f)

    with open(os.path.join(folder_name, "layers.txt"), "w") as f:
        f.write(model.layers_list())


    X_train, X_valid = train_test_split(data, test_size=valid_split, random_state=42)

    #loss = nn.MSELoss()
    loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    validation_losses = []
    for epoch in range(1, epochs+1):
        model.train()
        training_loss_batch = []
        for batch in chunker(X_train, batch_size=batch_size):
            batch = torch.Tensor(batch)
            prediction = model(batch)
            error = loss(prediction, batch)
            training_loss_batch.append(error.item())

            error.backward()
            optimizer.step()
            optimizer.zero_grad()
        training_loss = np.mean(training_loss_batch)
        writer.add_scalar('Loss/train', training_loss, epoch)

        model.eval()
        validation_loss_batch = []
        for batch in chunker(X_valid, batch_size=batch_size):
            batch = torch.Tensor(batch)
            prediction = model(batch)
            error = loss(prediction, batch)
            validation_loss_batch.append(error.item())
        validation_loss = np.mean(validation_loss_batch)
        validation_losses.append(validation_loss)
        writer.add_scalar('Loss/valid', validation_loss, epoch)

        print("Epoch {} - Train Loss {} - Valid Loss {}".format(epoch, training_loss, validation_loss))
        if validation_loss == min(validation_losses):
            torch.save(model, os.path.join(folder_name, "model.pt"))

        if not min(validation_losses) == min(validation_losses[-patience:]):
            print("Early stopping..")
            break

###########################################################################################################

class VAE(nn.Module):
    def __init__(self, encoder_layers, split_layer, decoder_layers):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc1 = deepcopy(split_layer)
        self.fc2 = deepcopy(split_layer)
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h1 = self.encoder(torch.Tensor(x))
        mu = self.fc1(h1)
        logvar = self.fc2(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 385))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def layers_list(self):
        return ("Encoder: " + str(self.encoder) 
        + "\nSplit layer: " + str(self.fc1)
        + "\nDencoder: " + str(self.decoder))

# Reconstruction + KL divergence losses summed over all elements and batch
def VAE_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    L1 = nn.L1Loss()(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

class VAE_loss(nn.Module):
    def __init__(self):
        super(VAE_loss, self).__init__()
        self.L1 = nn.L1Loss()

    def forward(self, recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        L1 = self.L1(recon_x, x) 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return L1 + KLD

def train_VAE(model, data, params):
    if not os.path.exists("trained_models"):
        os.mkdir("trained_models")
    folder_name = os.path.join("trained_models", "VAE", time.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(folder_name)


    lr = params.get("lr", 1e-3)
    params["lr"] = lr
    wd = params.get("wd", 1e-4)
    params["wd"] = wd
    epochs = params.get("epochs", 20)
    params["epochs"] = epochs
    patience = params.get("patience", 3)
    params["patience"] = patience
    valid_split = params.get("valid_split", 0.1)
    params["valid_split"] = valid_split
    batch_size = params.get("batch_size", 64)
    params["batch_size"] = batch_size

    with open(os.path.join(folder_name, "params.json"), "w") as f:
        json.dump(params, f)

    with open(os.path.join(folder_name, "layers.txt"), "w") as f:
        f.write(model.layers_list())


    X_train, X_valid = train_test_split(data, test_size=valid_split, random_state=42)

    #loss = nn.MSELoss()
    loss = VAE_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    validation_losses = []
    for epoch in range(1, epochs+1):
        model.train()
        training_loss_batch = []
        for batch in chunker(X_train, batch_size=batch_size):
            batch = torch.Tensor(batch)
            prediction, mu, logvar = model(batch)
            error = loss(prediction, batch, mu, logvar)
            training_loss_batch.append(error.item())

            error.backward()
            optimizer.step()
            optimizer.zero_grad()
        training_loss = np.mean(training_loss_batch)
        writer.add_scalar('Loss/train', training_loss, epoch)

        model.eval()
        validation_loss_batch = []
        for batch in chunker(X_valid, batch_size=batch_size):
            batch = torch.Tensor(batch)
            prediction, mu, logvar = model(batch)
            error = loss(prediction, batch, mu, logvar)
            validation_loss_batch.append(error.item())
        validation_loss = np.mean(validation_loss_batch)
        validation_losses.append(validation_loss)
        writer.add_scalar('Loss/valid', validation_loss, epoch)

        print("Epoch {} - Train Loss {} - Valid Loss {}".format(epoch, training_loss, validation_loss))
        if validation_loss == min(validation_losses):
            torch.save(model, os.path.join(folder_name, "model.pt"))

        if not min(validation_losses) == min(validation_losses[-patience:]):
            print("Early stopping..")
            break
