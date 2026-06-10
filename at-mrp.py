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
# 1. Model Definition
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
# 2. Attack Algorithms
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

# ==========================================
# 3. Training Loops with Progress Tracking
# ==========================================
def train(model, device, train_loader, optimizer, epochs=3):
    model.train()
    print("\n--- Starting Standard Training ---")
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 200 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] \tLoss: {loss.item():.6f}")

def fgsm_adversarial_train(model, device, train_loader, optimizer, epsilon, epochs=3):
    model.train()
    print("\n--- Starting FGSM Adversarial Training ---")
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            data.requires_grad = True
            output = model(data)
            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            adv_data = fgsm_attack(data, epsilon, data_grad)
            
            optimizer.zero_grad()
            output_adv = model(adv_data)
            loss_adv = F.nll_loss(output_adv, target)
            loss_adv.backward()
            optimizer.step()
            
            if batch_idx % 200 == 0:
                print(f"FGSM AT Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] \tLoss: {loss_adv.item():.6f}")

def pgd_adversarial_train(model, device, train_loader, optimizer, epsilon, epochs=3):
    print("\n--- Starting PGD Adversarial Training (Madry's Defense) ---")
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # FIX 1: Set to eval mode to get deterministic gradients for attack generation
            model.eval() 
            
            # FIX 2: Increase alpha so 7 iterations can reach the 0.3 boundary
            # 7 steps * 0.06 = 0.42 (safely covers the 0.3 epsilon ball)
            adv_data = pgd_attack(model, data, target, epsilon, alpha=0.06, iters=7)
            
            # FIX 3: Switch back to train mode for the actual model weight update
            model.train()
            
            optimizer.zero_grad()
            output_adv = model(adv_data)
            loss_adv = F.nll_loss(output_adv, target)
            loss_adv.backward()
            optimizer.step()
            
            if batch_idx % 200 == 0:
                print(f"PGD AT Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] \tLoss: {loss_adv.item():.6f}")

# ==========================================
# 4. Comprehensive Evaluation
# ==========================================
def evaluate_model(model, model_name, device, test_loader, epsilon):
    model.eval()
    correct_clean = 0
    total_samples = 0
    
    fgsm_success = 0
    fgsm_blur_success = 0
    pgd_success = 0
    pgd_blur_success = 0
    
    print(f"\n--- Evaluating {model_name} ---")
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        total_samples += len(data)
        
        # Clean Eval
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] 
        correct_mask = (init_pred.flatten() == target).flatten()
        correct_clean += correct_mask.sum().item()
        
        if correct_mask.sum() == 0:
            continue
            
        data_c = data[correct_mask]
        target_c = target[correct_mask]
        
        # --- FGSM Attack ---
        data_c.requires_grad = True
        out_c = model(data_c)
        loss = F.nll_loss(out_c, target_c)
        model.zero_grad()
        loss.backward()
        data_grad = data_c.grad.data
        
        adv_fgsm = fgsm_attack(data_c, epsilon, data_grad)
        pred_fgsm = model(adv_fgsm).max(1, keepdim=True)[1].flatten()
        fgsm_success += (pred_fgsm != target_c).sum().item()
        
        blurred_fgsm = TF.gaussian_blur(adv_fgsm, kernel_size=[3, 3], sigma=[1.0, 1.0])
        pred_blur_fgsm = model(blurred_fgsm).max(1, keepdim=True)[1].flatten()
        fgsm_blur_success += (pred_blur_fgsm != target_c).sum().item()
        
        # --- PGD Attack ---
        adv_pgd = pgd_attack(model, data_c, target_c, epsilon, alpha=0.01, iters=40)
        pred_pgd = model(adv_pgd).max(1, keepdim=True)[1].flatten()
        pgd_success += (pred_pgd != target_c).sum().item()
        
        blurred_pgd = TF.gaussian_blur(adv_pgd, kernel_size=[3, 3], sigma=[1.0, 1.0])
        pred_blur_pgd = model(blurred_pgd).max(1, keepdim=True)[1].flatten()
        pgd_blur_success += (pred_blur_pgd != target_c).sum().item()

    clean_acc = correct_clean / total_samples
    asr_fgsm = fgsm_success / correct_clean
    asr_blur_fgsm = fgsm_blur_success / correct_clean
    asr_pgd = pgd_success / correct_clean
    asr_blur_pgd = pgd_blur_success / correct_clean
    
    print(f"Clean Accuracy: {clean_acc*100:.2f}%")
    print(f"Standard FGSM ASR: {asr_fgsm*100:.2f}%")
    print(f"FGSM ASR (with Easy Defense - Blur): {asr_blur_fgsm*100:.2f}%")
    print(f"Standard PGD ASR: {asr_pgd*100:.2f}%")
    print(f"PGD ASR (with Easy Defense - Blur): {asr_blur_pgd*100:.2f}%")

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 1. Standard Model
    std_model = Net().to(device)
    optimizer_std = optim.Adam(std_model.parameters(), lr=1e-3)
    train(std_model, device, train_loader, optimizer_std, epochs=3)
    evaluate_model(std_model, "Standard Model", device, test_loader, epsilon=0.3)
    
    # 2. FGSM Adversarially Trained Model
    fgsm_model = Net().to(device)
    optimizer_fgsm = optim.Adam(fgsm_model.parameters(), lr=1e-3)
    fgsm_adversarial_train(fgsm_model, device, train_loader, optimizer_fgsm, epsilon=0.3, epochs=3)
    evaluate_model(fgsm_model, "FGSM-Trained Model", device, test_loader, epsilon=0.3)

    # 3. PGD Adversarially Trained Model (Madry's Defense)
    pgd_model = Net().to(device)
    optimizer_pgd = optim.Adam(pgd_model.parameters(), lr=1e-3)
    pgd_adversarial_train(pgd_model, device, train_loader, optimizer_pgd, epsilon=0.3, epochs=3)
    evaluate_model(pgd_model, "PGD-Trained Model", device, test_loader, epsilon=0.3)
