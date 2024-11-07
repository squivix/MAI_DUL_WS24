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
#     test_dataset = TensorDataset(x_test)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#
#     # x_test = x_test.to(device)
#
#     model = IGPTModel(3, max_sequence_length=image_shape[0] * image_shape[1] + 1)
#     model.to(device)
#
#     optimizer = Adam(model.parameters(), lr=learning_rate)
#     train_losses = []
#     test_losses = []
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
#             batch_test_loss = np.empty(len(batches))
#             for i, [test_batch_x] in enumerate(test_loader):
#                 test_batch_x = test_batch_x.to(device)
#                 test_logits = model.forward(test_batch_x)
#                 test_loss = model.loss_function(test_logits, test_batch_x)
#                 batch_test_loss[i] = test_loss.item()
#             test_losses.append(batch_test_loss.mean())
#         model.train()
#         print(f"epoch {epoch + 1}/{max_epochs}, train loss: {train_losses[-1]}, test loss: {test_losses[-1]}")
#     samples = model.generate(100, device).unflatten(1, (image_shape[0], image_shape[1])).unsqueeze(-1).numpy(force=True)
#     return train_losses, test_losses, samples
#
#
# def main():
#     q3ab_save_results(1, 'a', q3_a)
#
#
# main()

import numpy as np
import torch
from deepul.hw1_helper import q5a_save_results
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from models.GATModel import GATModel


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

    learning_rate = 0.001
    max_epochs = 100
    context_length = 128
    sliding_window_step = batch_size

    vocabulary = ["<bos>", *list(set("".join(train_text + test_text))), "<eos>"]
    vocab_to_index = {value: index for index, value in enumerate(vocabulary)}

    x_train = [item for text in train_text for item in [0, *[vocab_to_index[c] for c in text], len(vocabulary) - 1]]
    x_test = [item for text in test_text for item in [0, *[vocab_to_index[c] for c in text], len(vocabulary) - 1]]
    x_train = torch.tensor(x_train).unfold(0, context_length, sliding_window_step)
    x_test = torch.tensor(x_test).unfold(0, context_length, sliding_window_step)

    train_dataset = TensorDataset(x_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GATModel(vocab_size=len(vocabulary), context_length=context_length, embed_dim=128)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []
    for epoch in range(max_epochs):
        batches = iter(train_loader)

        batch_train_loss = np.empty(len(batches))
        for i, [batch_x] in (enumerate(batches)):
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
            batch_test_loss = np.empty(len(batches))
            for i, [test_batch_x] in enumerate(test_loader):
                test_batch_x = test_batch_x.to(device)
                test_logits = model.forward(test_batch_x)
                test_loss = model.loss_function(test_logits, test_batch_x)
                batch_test_loss[i] = test_loss.item()
            test_losses.append(batch_test_loss.mean())
        model.train()
        print(f"epoch {epoch + 1}/{max_epochs}, train loss: {train_losses[-1]}, test loss: {test_losses[-1]}")
    samples = model.generate(5, max_sequence_length=128, device=device).numpy(force=True)
    text_samples = []
    for sample in samples:
        gen_text = []
        for vi in sample:
            gen_text.append(vocabulary[vi])
            if vi == len(vocabulary) - 1:
                break
        text_samples.append("".join(gen_text))
    return train_losses, test_losses, text_samples


def main():
    q5a_save_results(q5_a)


main()
