# ================== Google Drive Mount ==================
from google.colab import drive
drive.mount('/content/drive')

# ================== Imports ==================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.notebook import tqdm
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ================== Paths ==================
drive_target_folder = "/content/drive/MyDrive/SynthRAD2023_Task1"
root_dir = os.path.join(drive_target_folder, "Task1")
checkpoint_dir = os.path.join(drive_target_folder, "checkpoints_light_cbam_128")
os.makedirs(checkpoint_dir, exist_ok=True)

# ================== Dataset ==================
class BrainMRICTDataset(Dataset):
    def __init__(self, patient_dirs, target_shape=(128,128,128), device='cpu'):
        self.patient_dirs = patient_dirs
        self.target_shape = tuple(target_shape)
        self.device = device

    def __len__(self):
        return len(self.patient_dirs)

    def load_nii(self, path):
        return nib.load(path).get_fdata().astype(np.float32)

    def z_score_normalize(self, volume):
        m = volume.mean(); s = volume.std()
        return (volume - m) / (s + 1e-8)

    def clip_and_scale_ct(self, ct, min_hu=-1000, max_hu=2000):
        ct = np.clip(ct, min_hu, max_hu)
        ct = 2 * (ct - min_hu) / (max_hu - min_hu) - 1
        return ct

    def crop_or_pad_center(self, vol):
        target = self.target_shape
        cur = vol.shape
        pad = []
        for c, t in zip(cur[::-1], target[::-1]):
            if c < t:
                total = t - c; before = total // 2; after = total - before
                pad.append((before, after))
            else:
                pad.append((0,0))
        pad = pad[::-1]
        if any(p != (0,0) for p in pad):
            vol = np.pad(vol, pad_width=pad, mode='constant', constant_values=0)
            cur = vol.shape
        start = [(c - t)//2 for c,t in zip(cur, target)]
        end = [start[i] + target[i] for i in range(3)]
        return vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    def __getitem__(self, idx):
        folder = self.patient_dirs[idx]
        mr_path = os.path.join(folder, 'mr.nii.gz')
        ct_path = os.path.join(folder, 'ct.nii.gz')
        mr = self.load_nii(mr_path); ct = self.load_nii(ct_path)
        mr = self.z_score_normalize(mr)
        ct = self.clip_and_scale_ct(ct)
        mr = self.crop_or_pad_center(mr)
        ct = self.crop_or_pad_center(ct)
        mr_t = torch.tensor(mr, dtype=torch.float32).unsqueeze(0)
        ct_t = torch.tensor(ct, dtype=torch.float32).unsqueeze(0)
        return mr_t.to(self.device), ct_t.to(self.device)

# ================== CBAM 3D ==================
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, max(1, in_planes//ratio), kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(1, in_planes//ratio), in_planes, kernel_size=1, bias=True)
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        a = self.fc(self.avg_pool(x))
        m = self.fc(self.max_pool(x))
        return self.sig(a + m) * x

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1)//2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, maxv], dim=1)
        out = self.conv(cat)
        return self.sig(out) * x

class CBAM3D(nn.Module):
    def __init__(self, channels, ratio=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention3D(channels, ratio)
        self.sa = SpatialAttention3D(kernel_size)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ================== UNet3D + CBAM ==================
class UNet3D_CBAM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[8,16,32,64]):
        super().__init__()
        self.encoder = nn.ModuleList()
        for f in features:
            self.encoder.append(self._block(in_channels, f))
            in_channels = f
        self.bottleneck = self._block(features[-1], features[-1]*2)
        self.attentions = nn.ModuleList([CBAM3D(f) for f in features])
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        prev = features[-1]*2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(prev, f, kernel_size=2, stride=2))
            self.decoder.append(self._block(f*2, f))
            prev = f
        self.final = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            att = self.attentions[i](x)
            skips.append(att)
            x = F.max_pool3d(x, 2)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                diffZ = skip.size(2) - x.size(2)
                diffY = skip.size(3) - x.size(3)
                diffX = skip.size(4) - x.size(4)
                x = F.pad(x, [diffX//2, diffX-diffX//2,
                              diffY//2, diffY-diffY//2,
                              diffZ//2, diffZ-diffZ//2])
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[i](x)
        return self.final(x)

# ================== Dataset split ==================
all_patients = []
for root, dirs, files in os.walk(root_dir):
    if 'mr.nii.gz' in files and 'ct.nii.gz' in files:
        all_patients.append(root)

print(f"Found patients: {len(all_patients)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = BrainMRICTDataset(all_patients, target_shape=(128,128,128), device=device)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ================== Model and Training ==================
model = UNet3D_CBAM().to(device)

def combined_loss(pred, target):
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    return 0.5*mse + 0.5*l1

criterion = combined_loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

num_epochs = 50
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train(); train_loss = 0
    for mri, ct in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(mri)
            loss = criterion(out, ct)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # ==== Validation ====
    model.eval(); val_loss = 0; mae_list=[]; psnr_list=[]; ssim_list=[]
    with torch.no_grad():
        for mri, ct in val_loader:
            with torch.cuda.amp.autocast():
                out = model(mri)
                loss = criterion(out, ct).item()
            val_loss += loss

            out_denorm = (out + 1) / 2
            ct_denorm = (ct + 1) / 2
            mae_list.append(F.l1_loss(out_denorm, ct_denorm).item())

            out_np = out_denorm[0].cpu().numpy()
            ct_np = ct_denorm[0].cpu().numpy()
            psnr_list.append(peak_signal_noise_ratio(ct_np, out_np, data_range=1))
            ssim_list.append(structural_similarity(ct_np.squeeze(), out_np.squeeze(), data_range=1))

    val_loss /= len(val_loader)
    mae = np.mean(mae_list); psnr = np.mean(psnr_list); ssim = np.mean(ssim_list)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
          f"MAE={mae:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
        print("✅ Saved new best model")

    if (epoch+1) % 5 == 0:
        mri, ct = next(iter(val_loader))
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = model(mri)
        out_denorm = (out + 1) / 2
        ct_denorm = (ct + 1) / 2
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow(mri[0,0,:,:,64].cpu(), cmap="gray"); plt.title("Input MRI")
        plt.subplot(1,3,2); plt.imshow(ct_denorm[0,0,:,:,64].cpu(), cmap="gray"); plt.title("Ground Truth CT")
        plt.subplot(1,3,3); plt.imshow(out_denorm[0,0,:,:,64].cpu(), cmap="gray"); plt.title("Predicted CT")
        plt.show()

# ================== Testing ==================
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pt"), map_location=device))
model.eval()

patient_results = []
with torch.no_grad():
    for i, (mri, ct) in enumerate(test_loader):
        out = model(mri)
        out_denorm = (out + 1) / 2
        ct_denorm = (ct + 1) / 2
        loss = criterion(out_denorm, ct_denorm).item()
        mae = F.l1_loss(out_denorm, ct_denorm).item()
        out_np = out_denorm[0].cpu().numpy()
        ct_np = ct_denorm[0].cpu().numpy()
        psnr = peak_signal_noise_ratio(ct_np, out_np, data_range=1)
        ssim = structural_similarity(ct_np.squeeze(), out_np.squeeze(), data_range=1)
        orig_idx = test_dataset.indices[i]
        patient_id = os.path.basename(dataset.patient_dirs[orig_idx])
        patient_results.append({
            'Patient ID': patient_id,
            'Loss': loss,
            'MAE': mae,
            'PSNR': psnr,
            'SSIM': ssim
        })

df = pd.DataFrame(patient_results)
df.loc['Average'] = df.mean(numeric_only=True)
print(df)
csv_path = os.path.join(checkpoint_dir, "test_results.csv")
df.to_csv(csv_path, index=False)
print(f"📊 Results saved to {csv_path}")
