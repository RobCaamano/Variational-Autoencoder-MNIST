import warnings
warnings.filterwarnings("ignore")

from sklearn.manifold import TSNE
from torchvision import datasets, transforms
import numpy as np
import pathlib
import torch

import matplotlib.pyplot as plt
import seaborn as sns

# Architectures
from train_variational_autoencoder import Model, VariationalMeanField

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config class containing best hyperparameters (model trained with these)
class Config:
    def __init__(self, **kwargs):
        self.latent_size = kwargs.get('latent_size', 128)
        self.variational = kwargs.get('variational', 'mean-field')
        self.flow_depth = kwargs.get('flow_depth', 4)
        self.data_size = kwargs.get('data_size', 784)
        self.learning_rate = kwargs.get('learning_rate', 0.0001)
        self.batch_size = kwargs.get('batch_size', 128)
        self.test_batch_size = kwargs.get('test_batch_size', 64)
        self.max_iterations = kwargs.get('max_iterations', 30000)
        self.log_interval = kwargs.get('log_interval', 1000)
        self.n_samples = kwargs.get('n_samples', 1000)
        self.use_gpu = kwargs.get('use_gpu', True)
        self.seed = kwargs.get('seed', 582838)
        self.train_dir = kwargs.get('train_dir', pathlib.Path('/tmp'))
        self.data_dir = kwargs.get('data_dir', pathlib.Path('/tmp'))

# Plot latent space of test data
def plot_latent_space(variational, test_loader):
    latent_vars = []
    labels = []
    for data, target in test_loader:
        data = data.to(device)

        # Get latent vars
        z, _ = variational(data.view(-1, 28*28))
        z = z.detach().cpu().numpy()
        z = np.squeeze(z)

        z_1 = z[:, 0]
        z_2 = z[:, 1]

        latent_vars.extend(np.stack((z_1, z_2), axis=1))
        labels.extend(target.numpy())

    latent_vars = np.array(latent_vars)
    labels = np.array(labels)

    print("Latent vars shape:", latent_vars.shape)
    print("Labels shape:", labels.shape)

    # Reduce dimensionality of latent vars to 2D
    t_SNE = TSNE(
        n_components=2, 
        perplexity=30, 
        learning_rate=100,
        verbose=1,
        n_jobs=-1,
        random_state=0
    )

    t_SNE_res = t_SNE.fit_transform(latent_vars)

    # Plot latent space
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x=t_SNE_res[:, 0], 
        y=t_SNE_res[:, 1],
        hue=labels,
        palette=sns.color_palette("hls", 10),
        legend="full",
        alpha=0.3
    )
    plt.xlabel("First dimension of sampled latent variable z_1")
    plt.ylabel("Second dimension of sampled latent variable z_2")
    plt.title("Final Latent Space")
    plt.show()


# Generate imgs of digits 0-9 from latent variables
def gen_imgs(model, variational, test_loader):
    digit_imgs = {}
    generated_imgs = {}

    # Iterate to collect one img / digit
    for batch in test_loader:
        x, labels = batch
        x = x.view(x.size(0), -1).to(device) # Flatten

        for i in range(len(labels)):
            label = labels[i].item()
            if label not in digit_imgs and label <= 9: # 0-9
                # Get img from dataset
                digit_imgs[label] = x[i].cpu().view(28, 28).numpy()
                with torch.no_grad():
                    # Generate img from latent variable
                    z, _ = variational(x[i].unsqueeze(0).to(device))
                    logits = model.generative_network(z)
                    reconstructed_image = torch.sigmoid(logits)
                    generated_imgs[label] = reconstructed_image.squeeze(0).cpu().view(28, 28).numpy()

            if len(digit_imgs) == 10:
                break
        if len(digit_imgs) == 10:
            break
    
    # Display digits 0-9 along w/ their generated counterparts
    fig, axes = plt.subplots(2, 10, figsize=(10, 2))
    for i, digit in enumerate(sorted(digit_imgs.keys())):
        # Plot dataset digits
        axes[0, i].imshow(digit_imgs[digit], cmap='gray')
        axes[0, i].set_title(str(digit))
        axes[0, i].axis('off')

        # Plot generated digits
        axes[1, i].imshow(generated_imgs[digit], cmap='gray')
        axes[1, i].axis('off')

    # Add text
    fig.text(0, 0.7, 'Dataset Digits', va='center', rotation='horizontal', fontsize=10)
    fig.text(0, 0.3, 'Generated Digits', va='center', rotation='horizontal', fontsize=10)

    plt.show()

def main():
    # Init config w/ best hyperparameters
    cfg = Config()

    # Load model & variational
    model = Model(
        latent_size=cfg.latent_size, 
        data_size=cfg.data_size)

    model.to(device)
    model_state = torch.load('model.pt')
    model.load_state_dict(model_state)  
    model.eval()

    variational = VariationalMeanField(
        latent_size=cfg.latent_size, 
        data_size=cfg.data_size
    )

    variational.to(device)
    variational_state = torch.load('variational.pt')
    variational.load_state_dict(variational_state)
    variational.eval()

    # Load data
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=True)

    # Generate digits 0-9
    gen_imgs(model, variational, test_loader)

    # Plot latent space
    plot_latent_space(variational, test_loader)

if __name__ == '__main__':
    main()