import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Model Definition (Unchanged)
# ==========================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ==========================================
# 2. Attack Algorithms (Unchanged)
# ==========================================
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def pgd_attack(model, image, label, epsilon, alpha, iters):
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach()
    for _ in range(iters):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = F.nll_loss(output, label)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image.detach() + alpha * data_grad.sign()
        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(original_image + eta, min=0, max=1).detach()
    return perturbed_image

def mifgsm_attack(model, image, label, epsilon, alpha, iters, decay=1.0):
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach()
    momentum = torch.zeros_like(image).detach()
    for _ in range(iters):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = F.nll_loss(output, label)
        model.zero_grad()
        loss.backward()
        grad = perturbed_image.grad.data
        grad_norm = torch.norm(grad, p=1)
        grad = grad / (grad_norm + 1e-8)
        momentum = decay * momentum + grad
        perturbed_image = perturbed_image.detach() + alpha * momentum.sign()
        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(original_image + eta, min=0, max=1).detach()
    return perturbed_image

# ==========================================
# 3. Training Loops (Standard & Adversarial)
# ==========================================
def train(model, device, train_loader, optimizer, epochs=3):
    model.train()
    print("--- Starting Standard Training ---")
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

def adversarial_train(model, device, train_loader, optimizer, epsilon, epochs=3):
    model.train()
    print("\n--- Starting Adversarial Training (FGSM) ---")
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Generate Adversarial Examples on the fly
            data.requires_grad = True
            output = model(data)
            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            adv_data = fgsm_attack(data, epsilon, data_grad)
            
            # Train on adversarial examples
            optimizer.zero_grad()
            output_adv = model(adv_data)
            loss_adv = F.nll_loss(output_adv, target)
            loss_adv.backward()
            optimizer.step()

# ==========================================
# 4. Visualization
# ==========================================
def generate_visual_evidence(model, device, test_loader, epsilon):
    model.eval()
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    
    # Get one correct prediction to visualize
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]
    
    # Pick the first image
    img = data[0:1]
    lbl = target[0:1]
    orig_pred = init_pred[0].item()
    
    img.requires_grad = True
    out = model(img)
    loss = F.nll_loss(out, lbl)
    model.zero_grad()
    loss.backward()
    
    adv_img = fgsm_attack(img, epsilon, img.grad.data)
    new_pred = model(adv_img).max(1, keepdim=True)[1].item()
    perturbation = adv_img - img
    
    # Plotting
    img_np = img.squeeze().detach().cpu().numpy()
    adv_np = adv_img.squeeze().detach().cpu().numpy()
    pert_np = perturbation.squeeze().detach().cpu().numpy()
    
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].imshow(img_np, cmap="gray")
    ax[0].set_title(f"Original\nPred: {orig_pred}")
    ax[0].axis('off')
    
    ax[1].imshow(pert_np, cmap="gray")
    ax[1].set_title("Perturbation")
    ax[1].axis('off')
    
    ax[2].imshow(adv_np, cmap="gray")
    ax[2].set_title(f"Adversarial\nPred: {new_pred}")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("adversarial_evidence.png")
    print("\n-> Saved visual evidence to 'adversarial_evidence.png' (Use this in Slide 6)")

# ==========================================
# 5. Evaluation & Defenses
# ==========================================
def test_attacks_and_defenses(model, device, test_loader, epsilon):
    model.eval()
    correct_clean = 0
    total_samples = 0
    
    # Trackers
    pgd_success = 0
    pgd_blur_success = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        total_samples += len(data)
        
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] 
        correct_mask = (init_pred.flatten() == target).flatten()
        correct_clean += correct_mask.sum().item()
        
        if correct_mask.sum() == 0:
            continue
            
        data_c = data[correct_mask]
        target_c = target[correct_mask]
        
        # Attack with PGD
        adv_pgd = pgd_attack(model, data_c, target_c, epsilon, alpha=0.01, iters=40)
        pred_pgd = model(adv_pgd).max(1, keepdim=True)[1].flatten()
        pgd_success += (pred_pgd != target_c).sum().item()
        
        # Apply Easy Defense (Gaussian Blur) to the adversarial images
        blurred_adv_pgd = TF.gaussian_blur(adv_pgd, kernel_size=[3, 3], sigma=[1.0, 1.0])
        pred_blur_pgd = model(blurred_adv_pgd).max(1, keepdim=True)[1].flatten()
        pgd_blur_success += (pred_blur_pgd != target_c).sum().item()

    clean_acc = correct_clean / total_samples
    asr_pgd = pgd_success / correct_clean
    asr_blur = pgd_blur_success / correct_clean
    
    print(f"Clean Accuracy: {clean_acc*100:.2f}%")
    print(f"Standard PGD ASR: {asr_pgd*100:.2f}%")
    print(f"PGD ASR (with Easy Defense - Blur): {asr_blur*100:.2f}%")

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 1. Train Standard Model
    std_model = Net().to(device)
    optimizer_std = optim.Adam(std_model.parameters(), lr=1e-3)
    train(std_model, device, train_loader, optimizer_std, epochs=3)
    
    # Generate Visuals for the slides
    generate_visual_evidence(std_model, device, test_loader, epsilon=0.3)
    
    print("\n--- Evaluating Standard Model ---")
    test_attacks_and_defenses(std_model, device, test_loader, epsilon=0.3)
    
    # 2. Train Robust Model (Adversarial Training)
    robust_model = Net().to(device)
    optimizer_rob = optim.Adam(robust_model.parameters(), lr=1e-3)
    adversarial_train(robust_model, device, train_loader, optimizer_rob, epsilon=0.3, epochs=3)
    
    print("\n--- Evaluating Adversarially Trained Model ---")
    # We test without the blur defense here to show the model's inherent robustness
    test_attacks_and_defenses(robust_model, device, test_loader, epsilon=0.3)