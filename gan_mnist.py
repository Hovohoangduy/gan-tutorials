import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import os

torch.manual_seed(1105)

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

## preparing the training data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
)

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size, shuffle=True
)

real_samples, mnist_labels = next(iter(train_loader))
# for i in range(16):
#     ax = plt.subplot(4, 4, i + 1)
#     plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output
    
discriminator = Discriminator().to(device)

class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output
    
generator = Generator().to(device)

# training gan models
lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr)

for epoch in range(1, num_epochs + 1):
    for n, (real_samples, mnist_labels) in enumerate(train_loader):
        # data for training the discriminator
        real_samples = real_samples.to(device)
        real_samples_labels = torch.ones((batch_size, 1)).to(device)
        latent_space_samples = torch.randn((batch_size, 100)).to(device)
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(device)
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
        # training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()
        # data training the generator
        latent_space_samples = torch.randn((batch_size, 100)).to(device)
        # training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # loss
        if n == batch_size:
            print(f"Epoch: {epoch} \n Loss D: {loss_discriminator} \n Loss G: {loss_generator}")
            # visualize
            generated_samples_show = generated_samples.cpu().detach()
            if epoch % 5 == 0:
                for i in range(16):
                    ax = plt.subplot(4, 4, i + 1)
                    plt.imshow(generated_samples_show[i].reshape(28, 28), cmap="gray_r")
                    plt.xticks([])
                    plt.yticks([])
                os.makedirs("tmp", exist_ok=True)
                plt.savefig(f"tmp/{epoch}.jpg")
                plt.close()

latent_space_samples = torch.randn(batch_size, 100).to(device)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.cpu().detach()
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
plt.show()

torch.cuda.empty_cache()