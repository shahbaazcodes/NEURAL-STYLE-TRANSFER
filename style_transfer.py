import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image size
img_size = 512 if torch.cuda.is_available() else 256

# Preprocessing transforms
loader = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# Normalization (as used during VGG training)
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def im_convert(tensor):
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = torch.clamp(image, 0, 1)
    return transforms.ToPILImage()(image)

# Content and style layers from VGG19
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Load images
content_img = load_image('content.jpg')
style_img = load_image('style.jpg')

assert content_img.size() == style_img.size(), "Style and content images must be the same size"

# Define content loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

# Define gram matrix for style
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

# Define style loss
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Load VGG model
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Normalization module
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std

# Build model
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                style_img, content_img):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # layer count
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim the model after last loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:i+1]

    return model, style_losses, content_losses

# Input image (starting point)
input_img = content_img.clone()

# Run style transfer
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print("Starting Style Transfer...")

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0.0
            content_score = 0.0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            loss = style_score * style_weight + content_score * content_weight
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}:")
                print(f"Style Loss : {style_score.item():.4f} Content Loss: {content_score.item():.4f}")

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

# Run and save result
output = run_style_transfer(cnn, normalization_mean, normalization_std,
                            content_img, style_img, input_img)

output_image = im_convert(output)
output_image.save('stylized_output.jpg')
print("Style transfer completed and image saved as 'stylized_output.jpg'")