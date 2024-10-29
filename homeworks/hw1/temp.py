from torch.optim import Adam
from models.IGPTModel import IGPTModel
import torch
from utils import rescale_tensor
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from deepul.hw1_helper import (
    # Q3
    q3ab_save_results,
    q3c_save_results,
)


def q3_a(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of color images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of color images with values in {0, 1}
    image_shape: (H, W, 1), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """
    batch_size = 3
    learning_rate = 0.001
    max_epochs = 100

    # rescale from 0/1 to -1/+1
    train_dataset = TensorDataset(
        rescale_tensor(torch.tensor(train_data).permute((0, 3, 1, 2)).float(), (0, 1), (-1, +1)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    x_test = rescale_tensor((torch.tensor(test_data).permute((0, 3, 1, 2)).float()), (0, 1), (-1, +1))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_test = x_test.to(device)

    model = IGPTModel()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []
    for epoch in range(max_epochs):
        batches = iter(train_loader)

        batch_train_loss = np.empty(len(batches))
        for i, [batch_x] in enumerate(batches):
            batch_x = batch_x.to(device)
            logits = model.forward(batch_x)

            loss = model.loss_function(batch_x, logits)
            batch_train_loss[i] = loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_losses.append(batch_train_loss.mean().item())

        model.eval()
        with torch.no_grad():
            test_logits = model.forward(x_test)
            test_loss = model.loss_function(x_test, test_logits)
            test_losses.append(test_loss.item())
        model.train()
        print(f"epoch {epoch + 1}/{max_epochs}, train loss: {train_losses[-1]}, test loss: {test_losses[-1]}")
    samples = model.generate(100, device).permute((0, 2, 3, 1)).numpy(force=True)
    return train_losses, test_losses, samples


def main():
    q3ab_save_results(1, 'a', q3_a)

main()