# Variational-Autoencoder-MNIST

train_variational_autoeconder.py: runs an optuna study to optimize hyperparameters and trains the best model

inference.py: runs inference on the trained VAE to generate the final latent space as well as digits.

## Training

### Loss Curves
![Loss Curves](https://github.com/RobCaamano/Variational-Autoencoder-MNIST/blob/main/img/Train%20%26%20Val%20Loss.png)

### Optuna Optimization Result
![Optuna Output](https://github.com/RobCaamano/Variational-Autoencoder-MNIST/blob/main/img/Optuna%20Output.png)

## Final Latent Variable Space

![Latent Space](https://github.com/RobCaamano/Variational-Autoencoder-MNIST/blob/main/img/Latent%20Space.png)

## VAE Generated Digits

![Digits](https://github.com/RobCaamano/Variational-Autoencoder-MNIST/blob/main/img/gen_imgs.png)
