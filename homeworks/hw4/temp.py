import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from deepul.hw4_helper import q2_save_results
from models.Diffusion2 import Diffusion2


def q2(train_data, test_data):
    """
    train_data: A (50000, 32, 32, 3) numpy array of images in [0, 1]
    test_data: A (10000, 32, 32, 3) numpy array of images in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 10 x 10 grid of generated samples. Each row represents 10 samples generated
      for a specific number of diffusion timesteps. Do this for 10 evenly logarithmically spaced integers
      1 to 512, i.e. np.power(2, np.linspace(0, 9, 10)).astype(int)
    """

    batch_size = 256
    max_epochs = 0
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_max = torch.tensor(train_data.min(axis=0), dtype=torch.float32).transpose(-1, 0).unsqueeze(0).to(device)
    train_min = torch.tensor(train_data.max(axis=0), dtype=torch.float32).transpose(-1, 0).unsqueeze(0).to(device)

    def normalize_transform(data):
        return (2 * (data - train_min) / (train_max - train_min)) - 1

    def denormalize_transform(data):
        return (data + 1) / 2 * (train_max - train_min) + train_min

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32).transpose(-1, 1))
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32).transpose(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = Diffusion2(in_channels=3, hidden_dims=[64, 128, 256, 512], blocks_per_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = [0]
    test_losses = [0]
    for epoch in range(max_epochs):
        print(f"{epoch+1}/{max_epochs}")
        for [batch_x] in tqdm(train_loader):
            optimizer.zero_grad()
            batch_x = normalize_transform(batch_x.to(device))
            t, x_t, epsilon = model.generate_noise_steps(batch_x, device=device)
            epsilon_hat = model.forward(x_t, t)
            loss = model.loss_function(epsilon_hat, epsilon)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            test_batch_losses = []
            for [batch_x] in test_loader:
                batch_x = normalize_transform(batch_x.to(device))
                t, x_t, epsilon = model.generate_noise_steps(batch_x, device=device)
                epsilon_hat = model.forward(x_t, t)
                loss = model.loss_function(epsilon_hat, epsilon)
                test_batch_losses.append(loss.item())
            test_losses.append(np.array(test_batch_losses).mean().item())
        model.train()
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    all_samples = []
    for num_steps in np.power(2, np.linspace(0, 9, 10)).astype(int):
        samples = denormalize_transform(model.generate(10, num_steps, device)).transpose(1, -1).numpy(force=True)
        all_samples.append(samples)
    all_samples = np.array(all_samples)
    return train_losses, test_losses, all_samples


def main():
    q2_save_results(q2)


main()
