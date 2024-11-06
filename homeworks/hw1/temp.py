# import numpy as np
# import torch
# from torch.optim import Adam
# from torch.utils.data import TensorDataset, DataLoader
#
# from deepul.hw1_helper import (
#     # Q3
#     q3ab_save_results,
# )
# from models.IGPTModel import IGPTModel
#
#
# def q3_a(train_data, test_data, image_shape, dset_id):
#     """
#     train_data: A (n_train, H, W, 1) uint8 numpy array of color images with values in {0, 1}
#     test_data: A (n_test, H, W, 1) uint8 numpy array of color images with values in {0, 1}
#     image_shape: (H, W, 1), height, width, and # of channels of the image
#     dset_id: An identifying number of which dataset is given (1 or 2). Most likely
#            used to set different hyperparameters for different datasets
#
#     Returns
#     - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
#     - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
#     - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
#     """
#     batch_size = 128
#
#     learning_rate = 0.001
#     max_epochs = 25
#
#     train_dataset = TensorDataset(torch.tensor(train_data).permute((0, 3, 1, 2)).float())
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
#     def transform(x):
#         x_flat = torch.flatten(x, start_dim=1)
#         return torch.cat((2 * torch.ones(x_flat.shape[0], 1).to(x_flat.device), x_flat), dim=1).long()
#
#     x_test = transform(torch.tensor(test_data).permute((0, 3, 1, 2)).float())
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     x_test = x_test.to(device)
#
#     model = IGPTModel(max_sequence_length=20 * 20 + 1)
#     model.to(device)
#
#     optimizer = Adam(model.parameters(), lr=learning_rate)
#     train_losses = [0]
#     test_losses = [0]
#     for epoch in range(max_epochs):
#         batches = iter(train_loader)
#
#         batch_train_loss = np.empty(len(batches))
#         for i, [batch_x] in enumerate(batches):
#             batch_x = transform(batch_x.to(device))
#             logits = model.forward(batch_x)
#
#             loss = model.loss_function(logits, batch_x)
#             batch_train_loss[i] = loss
#
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#         train_losses.append(batch_train_loss.mean().item())
#
#         model.eval()
#         with torch.no_grad():
#             test_logits = model.forward(x_test)
#             test_loss = model.loss_function(test_logits, x_test)
#             test_losses.append(test_loss.item())
#         model.train()
#         print(f"epoch {epoch + 1}/{max_epochs}, train loss: {train_losses[-1]}, test loss: {test_losses[-1]}")
#     samples = model.generate(100, device).permute((0, 2, 3, 1)).numpy(force=True)
#     return train_losses, test_losses, samples
#
#
# def main():
#     q3ab_save_results(1, 'a', q3_a)
#
#
# main()
import math

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from deepul.hw1_helper import q5a_save_results
from models.GATModel import GATModel


def create_non_overlapping_windows(tokens, context_size, padding):
    tokens_tensor = torch.tensor(tokens)
    padded_length = context_size * math.ceil(len(tokens) / context_size)
    return torch.cat((tokens_tensor, torch.full((padded_length - len(tokens),), padding)), dim=0).unflatten(0, (context_size, padded_length // context_size))


def q5_a(train_text, test_text):
    """
    train_text: list[str] Train text sequences.
    test_text: list[str] Test text sequences.

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a list of 5 (str), 5 generated samples from the model.
    """
    batch_size = 128

    learning_rate = 0.00001
    max_epochs = 50
    context_length = 128
    vocabulary = ["<bos>", *list(set("".join(train_text + test_text))), "<eos>"]
    vocab_to_index = {value: index + 1 for index, value in enumerate(vocabulary)}

    x_train = [item for text in train_text for item in [0, *[vocab_to_index[c] for c in text], len(vocabulary) - 1]]
    x_test = [item for text in train_text for item in [0, *[vocab_to_index[c] for c in text], len(vocabulary) - 1]]
    x_train = create_non_overlapping_windows(x_train, context_length, len(vocabulary) - 1).T
    x_test = create_non_overlapping_windows(x_test, context_length, len(vocabulary) - 1).T

    train_dataset = TensorDataset(x_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_test = x_test.to(device)

    model = GATModel(vocab_size=len(vocabulary))
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

            loss = model.loss_function(logits, batch_x)
            batch_train_loss[i] = loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_losses.append(batch_train_loss.mean().item())

        model.eval()
        with torch.no_grad():
            test_logits = model.forward(x_test)
            test_loss = model.loss_function(test_logits, x_test)
            test_losses.append(test_loss.item())
        model.train()
        print(f"epoch {epoch + 1}/{max_epochs}, train loss: {train_losses[-1]}, test loss: {test_losses[-1]}")
    samples = model.generate(5, max_sequence_length=128, device=device).numpy(force=True)
    text_samples = ["".join([vocabulary[vi] for vi in sample]) for sample in samples]
    return train_losses, test_losses, text_samples


def main():
    q5a_save_results(q5_a)


main()
