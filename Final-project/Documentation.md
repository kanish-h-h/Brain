
 Hr image 96x96
 mini batch of 16

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# Define the Generator model
class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16, num_features=64):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, num_features, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_residual_blocks)])

        self.upsample1 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.upsample2 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)

        self.conv2 = nn.Conv2d(num_features, 3, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        residual = out
        out = self.res_blocks(out)
        out = self.upsample1(out)
        out = self.pixel_shuffle1(out)
        out = self.upsample2(out)
        out = self.pixel_shuffle2(out)
        out = self.conv2(out)
        return self.tanh(out + residual)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def conv_block(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            conv_block(3, 64, stride=1),
            conv_block(64, 64, stride=2),
            conv_block(64, 128, stride=1),
            conv_block(128, 128, stride=2),
            conv_block(128, 256, stride=1),
            conv_block(256, 256, stride=2),
            conv_block(256, 512, stride=1),
            conv_block(512, 512, stride=2),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Define the VGG-based Content Loss
class VGGContentLoss(nn.Module):
    def __init__(self):
        super(VGGContentLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg = nn.Sequential(*list(vgg[:36])).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        return nn.functional.mse_loss(sr_features, hr_features)

# Training setup
def train_srgan(generator, discriminator, dataloader, num_epochs=100):
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

    # Losses
    adversarial_loss = nn.BCELoss()
    content_loss = VGGContentLoss()

    for epoch in range(num_epochs):
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs, hr_imgs = lr_imgs.cuda(), hr_imgs.cuda()

            # Train Discriminator
            d_optimizer.zero_grad()

            real_labels = torch.ones(hr_imgs.size(0), 1).cuda()
            fake_labels = torch.zeros(hr_imgs.size(0), 1).cuda()

            real_outputs = discriminator(hr_imgs)
            fake_outputs = discriminator(generator(lr_imgs).detach())

            d_loss_real = adversarial_loss(real_outputs, real_labels)
            d_loss_fake = adversarial_loss(fake_outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            fake_outputs = discriminator(generator(lr_imgs))
            adv_loss = adversarial_loss(fake_outputs, real_labels)
            cont_loss = content_loss(generator(lr_imgs), hr_imgs)

            g_loss = cont_loss + 1e-3 * adv_loss

            g_loss.backward()
            g_optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Dataset class for loading LR and HR images
class SRDataset(Dataset):
    def __init__(self, root_dir, transform_lr, transform_hr):
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        hr_img = self.transform_hr(img)
        lr_img = self.transform_lr(img)
        return lr_img, hr_img

# Main execution
if __name__ == "__main__":
    # Data loading
    transform_hr = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    transform_lr = transforms.Compose([
        transforms.Resize((24, 24)),
        transforms.ToTensor()
    ])

    dataset = SRDataset("/path/to/dataset", transform_lr, transform_hr)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Model initialization
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    # Train SRGAN
    train_srgan(generator, discriminator, dataloader)

```