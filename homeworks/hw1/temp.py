import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from deepul.hw1_helper import (
    # Q3
    q3ab_save_results,
)
from models.IGPTModel import IGPTModel


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
    batch_size = 128

    learning_rate = 0.001
    max_epochs = 20

    train_dataset = TensorDataset(torch.tensor(train_data).permute((0, 3, 1, 2)).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def transform(x):
        x_flat = torch.flatten(x, start_dim=1)
        return  torch.cat((2 * torch.ones(x_flat.shape[0], 1).to(x_flat.device), x_flat), dim=1).long()

    # x_test = transform(torch.tensor(test_data).permute((0, 3, 1, 2)).float())

    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # x_test = x_test.to(device)

    model = IGPTModel(max_sequence_length=401)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_losses = [0]
    test_losses = [0]
    for epoch in range(max_epochs):
        batches = iter(train_loader)

        batch_train_loss = np.empty(len(batches))
        for i, [batch_x] in enumerate(batches):
            batch_x = transform(batch_x.to(device))
            logits = model.forward(batch_x)

            loss = model.loss_function(logits, batch_x)
            batch_train_loss[i] = loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_losses.append(batch_train_loss.mean().item())

        model.eval()
        # with torch.no_grad():
        #     test_logits = model.forward(x_test)
        #     test_loss = model.loss_function(test_logits, x_test)
        #     test_losses.append(test_loss.item())
        model.train()
        print(f"epoch {epoch + 1}/{max_epochs}, train loss: {train_losses[-1]}, test loss: {test_losses[-1]}")
    # model = torch.load("model-tmp.pkl")
    # with torch.no_grad():
    #     test_logits = model.forward(x_test)
    #     test_loss = model.loss_function( test_logits,x_test)
    #     i=100
    #     predicted_0 = torch.max(torch.softmax(test_logits[i, :, :], dim=0), dim=0).values.long()[1:].reshape(20, 20).numpy(force=True)
    #     true_0 = x_test[i, 1:].reshape(20, 20).numpy(force=True)
    #
    #     array1 = predicted_0
    #     array2 = true_0
    #     # Set up a figure with two subplots side by side
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    #
    #     # Display the first array as an image
    #     ax1.imshow(array1, cmap='gray', interpolation='nearest')
    #     ax1.set_title("Array 1")
    #     ax1.axis('off')  # Hide axes for a cleaner look
    #
    #     # Display the second array as an image
    #     ax2.imshow(array2, cmap='gray', interpolation='nearest')
    #     ax2.set_title("Array 2")
    #     ax2.axis('off')  # Hide axes for a cleaner look
    #
    #     plt.show()
    #
    #     print(test_loss)
    samples = model.generate(100, device).permute((0, 2, 3, 1)).numpy(force=True)
    # return [0], [0], samples
    return train_losses, test_losses, samples


def main():
    q3ab_save_results(1, 'a', q3_a)


main()
