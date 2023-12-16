"""Train variational autoencoder on binary MNIST data."""

import numpy as np
import random
import time

import torch
import torch.utils
import torch.utils.data
from torch import nn

import data
import flow
import pathlib

import matplotlib.pyplot as plt
import optuna

class Model(nn.Module):
    """Variational autoencoder, parameterized by a generative network."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.register_buffer("p_z_loc", torch.zeros(latent_size))
        self.register_buffer("p_z_scale", torch.ones(latent_size))
        self.log_p_z = NormalLogProb()
        self.log_p_x = BernoulliLogProb()
        self.generative_network = NeuralNetwork(
            input_size=latent_size, output_size=data_size, hidden_size=latent_size * 2
        )

    def forward(self, z, x):
        """Return log probability of model."""
        log_p_z = self.log_p_z(self.p_z_loc, self.p_z_scale, z).sum(-1, keepdim=True)
        logits = self.generative_network(z)
        # unsqueeze sample dimension
        logits, x = torch.broadcast_tensors(logits, x.unsqueeze(1))
        log_p_x = self.log_p_x(logits, x).sum(-1, keepdim=True)
        return log_p_z + log_p_x


class VariationalMeanField(nn.Module):
    """Approximate posterior parameterized by an inference network."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.inference_network = NeuralNetwork(
            input_size=data_size,
            output_size=latent_size * 2,
            hidden_size=latent_size * 2,
        )
        self.log_q_z = NormalLogProb()
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        """Return sample of latent variable and log prob."""
        loc, scale_arg = torch.chunk(
            self.inference_network(x).unsqueeze(1), chunks=2, dim=-1
        )
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
        z = loc + scale * eps  # reparameterization
        log_q_z = self.log_q_z(loc, scale, z).sum(-1, keepdim=True)
        return z, log_q_z


class VariationalFlow(nn.Module):
    """Approximate posterior parameterized by a flow (https://arxiv.org/abs/1606.04934)."""

    def __init__(self, latent_size, data_size, flow_depth):
        super().__init__()
        hidden_size = latent_size * 2
        self.inference_network = NeuralNetwork(
            input_size=data_size,
            # loc, scale, and context
            output_size=latent_size * 3,
            hidden_size=hidden_size,
        )
        modules = []
        for _ in range(flow_depth):
            modules.append(
                flow.InverseAutoregressiveFlow(
                    num_input=latent_size,
                    num_hidden=hidden_size,
                    num_context=latent_size,
                )
            )
            modules.append(flow.Reverse(latent_size))
        self.q_z_flow = flow.FlowSequential(*modules)
        self.log_q_z_0 = NormalLogProb()
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        """Return sample of latent variable and log prob."""
        loc, scale_arg, h = torch.chunk(
            self.inference_network(x).unsqueeze(1), chunks=3, dim=-1
        )
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
        z_0 = loc + scale * eps  # reparameterization
        log_q_z_0 = self.log_q_z_0(loc, scale, z_0)
        z_T, log_q_z_flow = self.q_z_flow(z_0, context=h)
        log_q_z = (log_q_z_0 + log_q_z_flow).sum(-1, keepdim=True)
        return z_T, log_q_z


class NeuralNetwork(nn.Module):
    '''
    Simple feedforward neural network with ReLU activations.
    '''
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        modules = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, input):
        return self.net(input)


class NormalLogProb(nn.Module):
    '''
    Computes the log probability of z under a Normal distribution
    
    The computation uses the formula for the log pdf of a normal distribution.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, loc, scale, z):
        var = torch.pow(scale, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)


class BernoulliLogProb(nn.Module):
    '''
    Computes the log probability of a Bernoulli distribution.

    Uses binary cross-entropy with logits loss function, which
    is the negative log likelihood for a Bernoulli distribution.
    '''
    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, target):
        # bernoulli log prob is equivalent to negative binary cross entropy
        return -self.bce_with_logits(logits, target)


def cycle(iterable):
    '''
    Cycle through an iterable indefinitely.

    Continuously yields items from the iterable, restarting from the beginning
    once all items have been yielded. This is used for training loop.
    '''
    while True:
        for x in iterable:
            yield x


@torch.no_grad()
def evaluate(n_samples, model, variational, eval_data):
    '''
    Evaluate the VAE model on given evaluation data.
    
    Computes the ELBO and the log probability of the data.
    Uses importance sampling for approximate marginal likelihood estimation.
    Returns the average ELBO and log probability per data point.
    '''
    model.eval()
    total_log_p_x = 0.0
    total_elbo = 0.0
    for batch in eval_data:
        x = batch[0].to(next(model.parameters()).device)
        z, log_q_z = variational(x, n_samples)
        log_p_x_and_z = model(z, x)
        # importance sampling of approximate marginal likelihood with q(z)
        # as the proposal, and logsumexp in the sample dimension
        elbo = log_p_x_and_z - log_q_z
        log_p_x = torch.logsumexp(elbo, dim=1) - np.log(n_samples)
        # average over sample dimension, sum over minibatch
        total_elbo += elbo.cpu().numpy().mean(1).sum()
        # sum over minibatch
        total_log_p_x += log_p_x.cpu().numpy().sum()
    n_data = len(eval_data.dataset)
    return total_elbo / n_data, total_log_p_x / n_data

class Config:
    '''
    Config class for setting up VAE training parameters.
    
    Allows for dynamic setting of VAE hyperparameters and training options.
    Defaults are provided for each parameter.
    '''
    def __init__(self, **kwargs):
        self.latent_size = kwargs.get('latent_size', 128)
        self.variational = kwargs.get('variational', 'flow')
        self.flow_depth = kwargs.get('flow_depth', 2)
        self.data_size = kwargs.get('data_size', 784)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.batch_size = kwargs.get('batch_size', 128)
        self.test_batch_size = kwargs.get('test_batch_size', 512)
        self.max_iterations = kwargs.get('max_iterations', 30000)
        self.log_interval = kwargs.get('log_interval', 1000)
        self.n_samples = kwargs.get('n_samples', 1000)
        self.use_gpu = kwargs.get('use_gpu', True)
        self.seed = kwargs.get('seed', 582838)
        self.train_dir = kwargs.get('train_dir', pathlib.Path('/tmp'))
        self.data_dir = kwargs.get('data_dir', pathlib.Path('/tmp'))

def train(cfg):
    '''
    Trains a VAE with the given config.

    Sets the computation device, initializes the model and variational architectures,
    optimizes with RMSprop, and trains over a specified number of iterations.
    Evaluates the model on validation data during training and on test data after training.
    Implements early stopping if there's no improvement on validation ELBO.
    Returns the test ELBO for the objective function. Returns training and validation 
    losses, and model states for the main function.
    '''
    device = torch.device("cuda:0" if cfg.use_gpu else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Init model and variational
    model = Model(latent_size=cfg.latent_size, data_size=cfg.data_size)
    if cfg.variational == "flow":
        variational = VariationalFlow(
            latent_size=cfg.latent_size,
            data_size=cfg.data_size,
            flow_depth=cfg.flow_depth,
        )
    elif cfg.variational == "mean-field":
        variational = VariationalMeanField(
            latent_size=cfg.latent_size, data_size=cfg.data_size
        )
    else:
        raise ValueError(
            "Variational distribution not implemented: %s" % cfg.variational
        )

    model.to(device)
    variational.to(device)

    optimizer = torch.optim.RMSprop(
        list(model.parameters()) + list(variational.parameters()),
        lr=cfg.learning_rate,
        centered=True,
    )

    # Load data (binary MNIST)
    fname = cfg.data_dir / "binary_mnist.h5"
    if not fname.exists():
        print("Downloading binary MNIST data...")
        data.download_binary_mnist(fname)
    train_data, valid_data, test_data = data.load_binary_mnist(
        fname, cfg.batch_size, cfg.test_batch_size, cfg.use_gpu
    )

    best_valid_elbo = -np.inf
    num_no_improvement = 0
    train_ds = cycle(train_data)
    t0 = time.time()

    # Train & Validation Losses
    train_losses = []
    valid_losses = []

    # Training loop
    for step in range(cfg.max_iterations):
        batch = next(train_ds)
        x = batch[0].to(device)
        model.zero_grad()
        variational.zero_grad()
        z, log_q_z = variational(x, n_samples=1)
        log_p_x_and_z = model(z, x)
        # average over sample dimension
        elbo = (log_p_x_and_z - log_q_z).mean(1)
        # sum over batch dimension
        loss = -elbo.sum(0)
        loss.backward()
        optimizer.step()

        if step % cfg.log_interval == 0:
            t1 = time.time()
            examples_per_sec = cfg.log_interval * cfg.batch_size / (t1 - t0)
            with torch.no_grad():
                valid_elbo, valid_log_p_x = evaluate(
                    cfg.n_samples, model, variational, valid_data
                )
            print(
                f"Step {step:<10d}\t"
                f"Train ELBO estimate: {elbo.detach().cpu().numpy().mean():<5.3f}\t"
                f"Validation ELBO estimate: {valid_elbo:<5.3f}\t"
                f"Validation log p(x) estimate: {valid_log_p_x:<5.3f}\t"
                f"Speed: {examples_per_sec:<5.2e} examples/s"
            )
            if valid_elbo > best_valid_elbo:
                num_no_improvement = 0
                best_valid_elbo = valid_elbo
                states = {
                    "model": model.state_dict(),
                    "variational": variational.state_dict(),
                }
                torch.save(states, cfg.train_dir / "best_state_dict")
            else:
                num_no_improvement += 1

            # Append Losses
            train_losses.append(elbo.detach().cpu().numpy().mean())
            valid_losses.append(valid_elbo)

            if num_no_improvement >= 15:
                print("Early stopping")
                break

            t0 = t1

    # Evaluate on test data
    checkpoint = torch.load(cfg.train_dir / "best_state_dict")
    model.load_state_dict(checkpoint["model"])
    variational.load_state_dict(checkpoint["variational"])
    test_elbo, test_log_p_x = evaluate(cfg.n_samples, model, variational, test_data)
    print(
        f"Step {step:<10d}\t"
        f"Test ELBO estimate: {test_elbo:<5.3f}\t"
        f"Test log p(x) estimate: {test_log_p_x:<5.3f}\t"
    )

    print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")

    return test_elbo, train_losses, valid_losses, model, variational

def objective(trial):
    '''
    Objective function for hyperparameter optimization using Optuna.
    
    Optimizes VAE hyperparameters for best ELBO on test set. Parameters are
    latent size, variational approach, flow depth, learning rate, and batch sizes.
    Returns the test ELBO after training with the current config.
    '''

    # Hyperparameters to optimize
    cfg = Config(
        latent_size=trial.suggest_categorical('latent_size', [64, 128]),
        variational=trial.suggest_categorical('variational', ['flow', 'mean-field']),
        flow_depth=trial.suggest_categorical('flow_depth', [1, 2, 4]),
        data_size=784,
        learning_rate=trial.suggest_categorical('learning_rate', [1e-4, 1e-2]),
        batch_size=trial.suggest_categorical('batch_size', [64, 128]),
        test_batch_size=trial.suggest_categorical('test_batch_size', [64, 128]),
        max_iterations=15000,
        log_interval=1000,
        n_samples=1000,
        use_gpu=True,
        seed=582838,
        train_dir=pathlib.Path('/tmp'),
        data_dir=pathlib.Path('/tmp')
    )

    # Train w/ cfg
    test_elbo, _, _, _, _ = train(cfg)

    return test_elbo

if __name__ == "__main__":
    start_time = time.time()

    # Optuna study for hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("Best hyperparameters:", study.best_params)

    best_params = study.best_params

    # Set cfg with best hyperparameters from optuna
    cfg = Config(
        latent_size = best_params['latent_size'],
        variational = best_params['variational'],
        flow_depth =  best_params['flow_depth'],
        data_size = 784,
        learning_rate = best_params['learning_rate'],
        batch_size = best_params['batch_size'],
        test_batch_size = best_params['test_batch_size'],
        max_iterations = 30000,
        log_interval = 1000,
        n_samples = 1000,
        use_gpu = True,
        seed = 582838,
        train_dir = pathlib.Path('/tmp'),
        data_dir = pathlib.Path('/tmp')
    )

    # Train using cfg with best hyperparameters
    _, train_losses, valid_losses, model, variational = train(cfg)

    # Save Model & Variational
    torch.save(model.state_dict(), 'model.pt')
    torch.save(variational.state_dict(), 'variational.pt')

    # Train & Validation Losses Plots
    plt.plot(train_losses[1:], label='Training Loss')
    plt.plot(valid_losses[1:], label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()