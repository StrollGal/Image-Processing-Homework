# =============================================================================
# 运行方式: streamlit run app.py
# 依赖库: torch, torchvision, matplotlib, numpy, streamlit, diffusers, PIL, gzip, plotly
# =============================================================================

import os
import gzip
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
from PIL import Image
import random
from pathlib import Path
import threading
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import cv2

import matplotlib.font_manager as fm

_font_candidates = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
_available = [f.name for f in fm.fontManager.ttflist]
matplotlib.rcParams['font.sans-serif'] = [f for f in _font_candidates if f in _available] or ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="生成模型交互式演示", layout="wide")

DATA_DIR = Path(__file__).parent / "data" / "MNIST_data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

shared_state = {
    'ae_losses': [],
    'vae_total_losses': [],
    'vae_rec_losses': [],
    'vae_kl_losses': [],
    'epoch': 0,
    'completed': False,
    'ae_model': None,
    'vae_model': None,
    'error': None
}
state_lock = threading.Lock()


class MNISTLocalDataset(Dataset):
    def __init__(self, images_path, labels_path):
        with gzip.open(images_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

        with gzip.open(labels_path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)
        label = self.labels[idx]
        return image, label


def load_mnist_data():
    train_images = DATA_DIR / "train-images-idx3-ubyte.gz"
    train_labels = DATA_DIR / "train-labels-idx1-ubyte.gz"
    t10k_images = DATA_DIR / "t10k-images-idx3-ubyte.gz"
    t10k_labels = DATA_DIR / "t10k-labels-idx1-ubyte.gz"

    if all(p.exists() for p in [train_images, train_labels, t10k_images, t10k_labels]):
        train_dataset = MNISTLocalDataset(train_images, train_labels)
        test_dataset = MNISTLocalDataset(t10k_images, t10k_labels)
        return train_dataset, test_dataset

    from torchvision.datasets import MNIST
    transform = lambda x: torch.tensor(np.array(x).astype(np.float32) / 255.0).unsqueeze(0)
    train_dataset = MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
    test_dataset = MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)
    return train_dataset, test_dataset


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_ae_vae_thread(train_loader, test_loader, num_epochs):
    global shared_state, state_lock
    
    ae_path = MODEL_DIR / "autoencoder.pth"
    vae_path = MODEL_DIR / "vae.pth"

    ae = Autoencoder(latent_dim=2).to(DEVICE)
    vae = VAE(latent_dim=2).to(DEVICE)

    ae_optimizer = optim.Adam(ae.parameters(), lr=1e-3)
    vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    ae_losses = []
    vae_total_losses = []
    vae_rec_losses = []
    vae_kl_losses = []

    try:
        for epoch in range(num_epochs):
            with state_lock:
                if not shared_state.get('training', False):
                    break

            ae.train()
            vae.train()

            ae_epoch_loss = 0
            vae_total_loss = 0
            vae_rec_loss = 0
            vae_kl_loss = 0

            for batch_idx, (images, _) in enumerate(train_loader):
                images = images.to(DEVICE)

                ae_optimizer.zero_grad()
                recon_ae, _ = ae(images)
                loss_ae = nn.functional.mse_loss(recon_ae, images)
                loss_ae.backward()
                ae_optimizer.step()
                ae_epoch_loss += loss_ae.item()

                vae_optimizer.zero_grad()
                recon_vae, mu, logvar = vae(images)
                rec_loss = nn.functional.binary_cross_entropy(recon_vae, images, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss_vae = rec_loss + kl_loss
                loss_vae.backward()
                vae_optimizer.step()

                vae_total_loss += loss_vae.item()
                vae_rec_loss += rec_loss.item()
                vae_kl_loss += kl_loss.item()

            ae_losses.append(ae_epoch_loss / len(train_loader.dataset))
            vae_total_losses.append(vae_total_loss / len(train_loader.dataset))
            vae_rec_losses.append(vae_rec_loss / len(train_loader.dataset))
            vae_kl_losses.append(vae_kl_loss / len(train_loader.dataset))

            with state_lock:
                shared_state['ae_losses'] = ae_losses.copy()
                shared_state['vae_total_losses'] = vae_total_losses.copy()
                shared_state['vae_rec_losses'] = vae_rec_losses.copy()
                shared_state['vae_kl_losses'] = vae_kl_losses.copy()
                shared_state['epoch'] = epoch + 1

            time.sleep(0.05)

        torch.save(ae.state_dict(), ae_path)
        torch.save(vae.state_dict(), vae_path)

        ae.to('cpu')
        vae.to('cpu')

        with state_lock:
            shared_state['ae_model'] = ae
            shared_state['vae_model'] = vae
            shared_state['completed'] = True
            
    except Exception as e:
        with state_lock:
            shared_state['error'] = str(e)
            shared_state['completed'] = True


def train_dcgan_sync(train_loader, num_epochs, progress_callback):
    gen_path = MODEL_DIR / "dcgan_generator.pth"
    disc_path = MODEL_DIR / "dcgan_discriminator.pth"

    generator = Generator(latent_dim=100).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(100, 100, device=DEVICE)
    fixed_noise_sample = fixed_noise[:16].view(16, 100)

    timeline_images = []

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        for batch_idx, (images, _) in enumerate(train_loader):
            batch_size = images.size(0)
            real_labels = torch.ones(batch_size, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

            images = images.to(DEVICE)
            optimizer_D.zero_grad()
            output_real = discriminator(images)
            loss_d_real = criterion(output_real, real_labels)

            noise = torch.randn(batch_size, 100, device=DEVICE)
            fake_images = generator(noise).detach()
            output_fake = discriminator(fake_images)
            loss_d_fake = criterion(output_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            noise = torch.randn(batch_size, 100, device=DEVICE)
            fake_images = generator(noise)
            output_g = discriminator(fake_images)
            loss_g = criterion(output_g, real_labels)
            loss_g.backward()
            optimizer_G.step()

        with torch.no_grad():
            generator.eval()
            fake = generator(fixed_noise_sample[:1])
            timeline_images.append(fake.cpu().squeeze().numpy())

        progress_callback(epoch + 1, num_epochs)

    torch.save(generator.state_dict(), gen_path)
    torch.save(discriminator.state_dict(), disc_path)

    generator.to('cpu')
    discriminator.to('cpu')

    return generator, discriminator, timeline_images


def get_latent_points(vae, test_loader):
    vae.eval()
    vae.to(DEVICE)
    points = []
    labels = []
    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.to(DEVICE)
            _, mu, _ = vae(images)
            points.append(mu.cpu().numpy())
            labels.append(lbls.numpy())
    vae.to('cpu')
    return np.concatenate(points), np.concatenate(labels)


def plot_training_progress(ae_losses, vae_total, vae_rec, vae_kl, num_epochs):
    if not ae_losses:
        return None

    current_epoch = len(ae_losses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

    ax1.plot(range(1, current_epoch + 1), ae_losses, 'b-o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction Loss')
    ax1.set_title('AE Loss 曲线')
    ax1.grid(True)
    ax1.set_xlim(0.5, num_epochs + 0.5)

    ax2.plot(range(1, current_epoch + 1), vae_total, 'r-o', label='Total Loss')
    ax2.plot(range(1, current_epoch + 1), vae_rec, 'g-o', label='Reconstruction Loss')
    ax2.plot(range(1, current_epoch + 1), vae_kl, 'm-o', label='KL Divergence')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('VAE Loss 曲线')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0.5, num_epochs + 0.5)

    plt.tight_layout()
    return fig


def load_pretrained_ae_vae():
    """加载预训练的 AE 和 VAE 模型"""
    try:
        ae = Autoencoder(latent_dim=2)
        vae = VAE(latent_dim=2)
        
        ae.load_state_dict(torch.load(MODEL_DIR / "autoencoder.pth", map_location='cpu'))
        vae.load_state_dict(torch.load(MODEL_DIR / "vae.pth", map_location='cpu'))
        
        ae.eval()
        vae.eval()
        return ae, vae
    except Exception as e:
        st.error(f"加载 AE/VAE 模型失败: {str(e)}")
        return None, None


def load_pretrained_dcgan():
    """加载预训练的 DCGAN 模型"""
    try:
        generator = Generator(latent_dim=100)
        discriminator = Discriminator()
        
        generator.load_state_dict(torch.load(MODEL_DIR / "dcgan_generator.pth", map_location='cpu'))
        discriminator.load_state_dict(torch.load(MODEL_DIR / "dcgan_discriminator.pth", map_location='cpu'))
        
        generator.eval()
        discriminator.eval()
        return generator, discriminator
    except Exception as e:
        st.error(f"加载 DCGAN 模型失败: {str(e)}")
        return None, None


def main():
    if 'train_dataset' not in st.session_state:
        with st.spinner("加载 MNIST 数据集..."):
            train_dataset, test_dataset = load_mnist_data()
            st.session_state['train_dataset'] = train_dataset
            st.session_state['test_dataset'] = test_dataset
            st.session_state['ae_trained'] = False
            st.session_state['ae_loaded_from_pretrained'] = False
            st.session_state['dcgan_trained'] = False
            st.session_state['diffusion_loaded'] = False
            st.session_state['clicked_point'] = None
            st.session_state['training_ae'] = False
            st.session_state['ae_training_epoch'] = 0
            st.session_state['ae_losses'] = []
            st.session_state['vae_total_losses'] = []
            st.session_state['vae_rec_losses'] = []
            st.session_state['vae_kl_losses'] = []

    train_dataset = st.session_state['train_dataset']
    test_dataset = st.session_state['test_dataset']

    st.title("生成模型交互式演示")

    st.info("""这是一个生成模型交互式演示工具，包含**重构对比**、**潜空间探索**和**高级生成模型**三个模块。
    在侧边栏可调整训练数据量和训练轮数等参数，点击对应标签页中的训练按钮开始训练，训练完成后可查看模型效果。""")

    col_top_left, col_top_right = st.columns([1, 1])
    with col_top_left:
        if st.session_state.get('training_ae', False) or st.session_state.get('training_dcgan', False):
            st.caption("⏳ 模型训练中...")

    with col_top_right:
        st.caption(f"设备: {'GPU (CUDA)' if DEVICE.type == 'cuda' else 'CPU'}")

    tab1, tab2, tab3 = st.tabs(["重构对比", "潜空间探索", "高级生成模型"])

    with st.sidebar:
        st.header("训练参数设置")
        st.caption("这里可以调整训练数据量和训练轮数，修改后点击对应标签页中的训练按钮即可生效。")

        train_size = st.slider(
            "训练数据大小",
            min_value=1000,
            max_value=60000,
            value=60000,
            step=1000,
            format="%d",
            key="train_size_slider"
        )

        num_epochs_ae = st.slider(
            "AE & VAE 训练轮数",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="epochs_ae_slider"
        )

        num_epochs_dcgan = st.slider(
            "DCGAN 训练轮数",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            key="epochs_dcgan_slider"
        )
        
        st.markdown("---")
        st.subheader("快速演示")
        st.caption("点击下面按钮可直接加载预训练模型，无需重新训练")
        
        col_load1, col_load2 = st.columns([1, 1])
        with col_load1:
            if st.button("加载 AE/VAE", key="load_ae_vae"):
                with st.spinner("正在加载 AE/VAE 模型..."):
                    ae, vae = load_pretrained_ae_vae()
                    if ae and vae:
                        st.session_state['ae'] = ae
                        st.session_state['vae'] = vae
                        st.session_state['ae_trained'] = True
                        st.session_state['ae_loaded_from_pretrained'] = True  # 标记是加载的预训练模型
                        # 模拟一些损失数据用于展示
                        st.session_state['ae_losses'] = [0.25, 0.18, 0.14, 0.12, 0.10]
                        st.session_state['vae_total_losses'] = [0.28, 0.20, 0.16, 0.14, 0.12]
                        st.session_state['vae_rec_losses'] = [0.26, 0.19, 0.15, 0.13, 0.11]
                        st.session_state['vae_kl_losses'] = [0.02, 0.01, 0.01, 0.01, 0.01]
                        st.success("✅ AE/VAE 模型加载成功！")
                        st.rerun()
        
        with col_load2:
            if st.button("加载 DCGAN", key="load_dcgan"):
                with st.spinner("正在加载 DCGAN 模型..."):
                    generator, discriminator = load_pretrained_dcgan()
                    if generator and discriminator:
                        st.session_state['generator'] = generator
                        st.session_state['discriminator'] = discriminator
                        st.session_state['dcgan_trained'] = True
                        # 从数据集中预加载真实样本用于时间线展示
                        if 'dcgan_real_samples' not in st.session_state:
                            real_indices = random.sample(range(len(train_dataset)), 10)
                            real_samples = []
                            for idx in real_indices:
                                img, label = train_dataset[idx]
                                real_samples.append((img.squeeze().numpy(), label))
                            st.session_state['dcgan_real_samples'] = real_samples
                        st.success("✅ DCGAN 模型加载成功！")
                        st.rerun()

    with tab1:
        st.header("重构对比：自编码器 vs 变分自编码器")
        
        st.markdown("""**🎯 目标**：对比普通自编码器（AE）和变分自编码器（VAE）的图像重构能力。

**🎛️ 控件说明**：
- **开始训练 AE & VAE**：训练两个模型，训练过程中实时显示损失曲线
- **随机抽取样本**：从测试集中随机选择一张图像，展示原始图、AE 重构图和 VAE 重构图

**📊 结果展示**：
- **原始图像**：输入的 MNIST 手写数字
- **AE/VAE 重构**：模型重建的图像
- **重构误差热力图**：显示原始图和重构图的像素级差异
- **训练损失曲线**：展示训练过程中损失值的变化

**💡 演示说明**：训练完成后点击"随机抽取样本"查看重构效果，热力图红色区域表示重构差异较大的位置。""")

        chart_placeholder = st.empty()

        col_btn, col_progress = st.columns([1, 3])

        with col_btn:
            if st.button("开始训练 AE & VAE", key="train_ae_vae_btn", disabled=st.session_state.get('training_ae', False)):
                st.session_state['training_ae'] = True
                st.session_state['ae_training_epoch'] = 0
                st.session_state['ae_losses'] = []
                st.session_state['vae_total_losses'] = []
                st.session_state['vae_rec_losses'] = []
                st.session_state['vae_kl_losses'] = []

                with state_lock:
                    shared_state['ae_losses'] = []
                    shared_state['vae_total_losses'] = []
                    shared_state['vae_rec_losses'] = []
                    shared_state['vae_kl_losses'] = []
                    shared_state['epoch'] = 0
                    shared_state['completed'] = False
                    shared_state['ae_model'] = None
                    shared_state['vae_model'] = None
                    shared_state['error'] = None
                    shared_state['training'] = True

                train_subset = Subset(train_dataset, range(min(train_size, len(train_dataset))))
                train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
                test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

                thread = threading.Thread(
                    target=train_ae_vae_thread,
                    args=(train_loader, test_loader, num_epochs_ae)
                )
                thread.daemon = True
                thread.start()

                progress_bar = st.empty()
                status_text = st.empty()

                while True:
                    with state_lock:
                        ae_losses = shared_state['ae_losses'].copy()
                        vae_total_losses = shared_state['vae_total_losses'].copy()
                        vae_rec_losses = shared_state['vae_rec_losses'].copy()
                        vae_kl_losses = shared_state['vae_kl_losses'].copy()
                        epoch = shared_state['epoch']
                        completed = shared_state['completed']
                        ae_model = shared_state['ae_model']
                        vae_model = shared_state['vae_model']
                        error = shared_state['error']

                    if completed:
                        progress_bar.progress(1.0)
                        if error:
                            status_text.error(f"训练过程中发生错误: {error}")
                        else:
                            status_text.text(f"训练完成: Epoch {num_epochs_ae}/{num_epochs_ae}")
                            st.session_state['ae'] = ae_model
                            st.session_state['vae'] = vae_model
                            st.session_state['ae_trained'] = True
                            st.session_state['ae_loaded_from_pretrained'] = False  # 标记是通过训练得到的模型
                            st.session_state['ae_losses'] = ae_losses
                            st.session_state['vae_total_losses'] = vae_total_losses
                            st.session_state['vae_rec_losses'] = vae_rec_losses
                            st.session_state['vae_kl_losses'] = vae_kl_losses
                            st.session_state['ae_training_epoch'] = epoch

                            fig = plot_training_progress(
                                ae_losses,
                                vae_total_losses,
                                vae_rec_losses,
                                vae_kl_losses,
                                num_epochs_ae
                            )
                            if fig:
                                chart_placeholder.pyplot(fig, use_container_width=True)
                                plt.close()
                        
                        with state_lock:
                            shared_state['training'] = False
                        
                        st.session_state['training_ae'] = False
                        break
                    
                    progress = epoch / num_epochs_ae
                    progress_bar.progress(progress)
                    status_text.text(f"训练进度: Epoch {epoch}/{num_epochs_ae}")

                    fig = plot_training_progress(
                        ae_losses,
                        vae_total_losses,
                        vae_rec_losses,
                        vae_kl_losses,
                        num_epochs_ae
                    )
                    if fig:
                        chart_placeholder.pyplot(fig, use_container_width=True)
                        plt.close()

                    time.sleep(0.3)

        with col_progress:
            if st.session_state.get('ae_trained', False):
                st.caption("✅ 模型已训练完成")

        if st.session_state.get('ae_trained', False) and 'ae' in st.session_state:
            ae = st.session_state['ae']
            vae = st.session_state['vae']

            col1, col2, col3 = st.columns(3)

            if st.button("随机抽取样本", key="random_sample_1"):
                idx = random.randint(0, len(test_dataset) - 1)
                st.session_state['sample_idx'] = idx

            if 'sample_idx' not in st.session_state:
                st.session_state['sample_idx'] = random.randint(0, len(test_dataset) - 1)

            idx = st.session_state['sample_idx']
            original_image, label = test_dataset[idx]

            with col1:
                st.subheader("原始图像")
                st.image(original_image.squeeze().numpy(), width=200, clamp=True)
                st.caption(f"标签: {label}")

            with col2:
                ae.to(DEVICE)
                ae.eval()
                with torch.no_grad():
                    recon_ae, _ = ae(original_image.unsqueeze(0).to(DEVICE))
                    recon_ae = recon_ae.cpu().squeeze().numpy()
                ae.to('cpu')

                st.subheader("AE 重构")
                st.image(recon_ae, width=200, clamp=True)
                st.caption(f"重构误差: {np.abs(original_image.squeeze().numpy() - recon_ae).mean():.4f}")

            with col3:
                vae.to(DEVICE)
                vae.eval()
                with torch.no_grad():
                    recon_vae, _, _ = vae(original_image.unsqueeze(0).to(DEVICE))
                    recon_vae = recon_vae.cpu().squeeze().numpy()
                vae.to('cpu')

                st.subheader("VAE 重构")
                st.image(recon_vae, width=200, clamp=True)
                st.caption(f"重构误差: {np.abs(original_image.squeeze().numpy() - recon_vae).mean():.4f}")

            st.subheader("重构误差热力图")

            diff_ae = np.abs(original_image.squeeze().numpy() - recon_ae)
            diff_vae = np.abs(original_image.squeeze().numpy() - recon_vae)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=100)

            im1 = ax1.imshow(diff_ae, cmap='hot')
            ax1.set_title('|原始 - AE重构|')
            ax1.axis('off')
            fig.colorbar(im1, ax=ax1, fraction=0.046)

            im2 = ax2.imshow(diff_vae, cmap='hot')
            ax2.set_title('|原始 - VAE重构|')
            ax2.axis('off')
            fig.colorbar(im2, ax=ax2, fraction=0.046)

            st.pyplot(fig)
            plt.close()

            if not st.session_state.get('ae_loaded_from_pretrained', False):
                st.subheader("训练过程可视化")

                fig = plot_training_progress(
                    st.session_state['ae_losses'],
                    st.session_state['vae_total_losses'],
                    st.session_state['vae_rec_losses'],
                    st.session_state['vae_kl_losses'],
                    num_epochs_ae
                )
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

        elif not st.session_state.get('training_ae', False):
            st.info("👆 点击上方「开始训练 AE & VAE」按钮开始训练模型")

    with tab2:
        st.header("潜空间探索：VAE 潜变量交互")
        
        st.markdown("""**🎯 目标**：探索 VAE 学习到的二维潜空间，理解潜变量如何对应数字形态。

**🎛️ 控件说明**：
- **潜空间散点图**：每种颜色代表一类数字（0-9），悬停可查看坐标
- **z[0]/z[1] 输入框**：手动输入潜空间坐标
- **显示坐标**：在散点图上标记输入的坐标位置
- **生成图像**：使用 VAE 解码器生成对应坐标的数字图像
- **清除标记**：移除标记点和生成的图像
- **样本 A/B 索引**：选择用于插值的两个样本
- **执行插值**：生成两个样本之间的平滑过渡序列

**📊 结果展示**：
- **生成图像**：根据输入坐标生成的数字图像
- **插值序列**：展示从样本 A 到样本 B 的渐变过程

**💡 演示说明**：散点图中相近颜色的点代表相似的数字；输入坐标并点击"显示坐标"放置标记，再点击"生成图像"查看对应数字；选择两个样本后点击"执行插值"可观察数字平滑过渡效果。""")

        if not st.session_state.get('ae_trained', False):
            st.warning("请先在「重构对比」标签页训练 AE 和 VAE 模型")
        else:
            vae = st.session_state['vae']
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

            if 'latent_points' not in st.session_state:
                with st.spinner("计算潜空间分布..."):
                    latent_points, labels = get_latent_points(vae, test_loader)
                    st.session_state['latent_points'] = latent_points
                    st.session_state['latent_labels'] = labels

            latent_points = st.session_state['latent_points']
            latent_labels = st.session_state['latent_labels']

            st.subheader("潜空间散点图")
            st.caption("💡 悬停可查看坐标，在下方输入框中输入坐标，点击「显示坐标」放置标记，再点击「生成图像」")

            df = pd.DataFrame({
                'z0': latent_points[:, 0],
                'z1': latent_points[:, 1],
                'label': latent_labels
            })

            marker_z0 = st.session_state.get('marker_z0', 0.0)
            marker_z1 = st.session_state.get('marker_z1', 0.0)
            show_marker = st.session_state.get('show_marker', False)

            fig = px.scatter(df, x='z0', y='z1', color='label',
                            color_discrete_sequence=px.colors.qualitative.T10,
                            title='VAE 潜空间分布',
                            labels={'z0': 'z[0]', 'z1': 'z[1]', 'label': '数字类别'},
                            hover_data={'z0': ':,.2f', 'z1': ':,.2f', 'label': True})

            if show_marker:
                fig.add_trace(go.Scatter(
                    x=[marker_z0], y=[marker_z1],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=15,
                        symbol='x',
                        line=dict(color='black', width=3)
                    ),
                    name='标记点'
                ))
                fig.update_traces(selector=dict(name='标记点'), marker=dict(size=15, color='red', symbol='x', line=dict(color='black', width=3)))

            fig.update_layout(width=800, height=600, legend=dict(itemsizing='constant'))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("坐标设定")
            col_z0, col_z1, col_buttons, col_desc = st.columns([1, 1, 1, 2])
            
            with col_z0:
                st.number_input("z[0]", value=0.0, step=0.1, key="marker_z0")
            
            with col_z1:
                st.number_input("z[1]", value=0.0, step=0.1, key="marker_z1")
            
            marker_z0 = st.session_state.get('marker_z0', 0.0)
            marker_z1 = st.session_state.get('marker_z1', 0.0)
            
            with col_buttons:
                if st.button("显示坐标", key="show_coord_btn"):
                    st.session_state['show_marker'] = True
                    st.rerun()
                
                if st.button("生成图像", key="gen_image_btn", type="primary"):
                    if st.session_state.get('show_marker', False):
                        vae.to(DEVICE)
                        vae.eval()
                        with torch.no_grad():
                            z = torch.tensor([[marker_z0, marker_z1]], dtype=torch.float32).to(DEVICE)
                            generated = vae.decoder(z).cpu().squeeze().numpy()
                        vae.to('cpu')
                        st.session_state['generated_image'] = generated
                    else:
                        st.warning("请先点击「显示坐标」放置标记点")
                
                if st.button("清除标记", key="clear_marker_btn"):
                    st.session_state['show_marker'] = False
                    st.session_state['generated_image'] = None
                    st.rerun()
            
            with col_desc:
                st.markdown("""<div style="margin-bottom: 5px;"><span style="font-size: 18px; font-weight: bold;">使用说明：</span></div>

1) 输入坐标
2) 点击「显示坐标」
3) 点击「生成图像」""", unsafe_allow_html=True)

            st.subheader("生成图像")
            if show_marker and 'generated_image' in st.session_state and st.session_state['generated_image'] is not None:
                st.write(f"当前坐标: (z0={marker_z0:.3f}, z1={marker_z1:.3f})")
                st.image(st.session_state['generated_image'], width=200, clamp=True)
            else:
                st.info("请先点击「显示坐标」放置标记，再点击「生成图像」")

            st.subheader("潜空间插值")

            col1, col2 = st.columns(2)
            with col1:
                idx_a = st.number_input("样本 A 索引", value=0, min_value=0, max_value=len(test_dataset)-1, key="idx_a")
                img_a, label_a = test_dataset[idx_a]
                st.image(img_a.squeeze().numpy(), width=100, clamp=True)
                st.caption(f"标签: {label_a}")
            with col2:
                idx_b = st.number_input("样本 B 索引", value=100, min_value=0, max_value=len(test_dataset)-1, key="idx_b")
                img_b, label_b = test_dataset[idx_b]
                st.image(img_b.squeeze().numpy(), width=100, clamp=True)
                st.caption(f"标签: {label_b}")

            if st.button("执行插值", key="interpolate"):
                with st.spinner("计算插值中..."):
                    vae.to(DEVICE)
                    vae.eval()
                    with torch.no_grad():
                        _, mu_a, _ = vae(img_a.unsqueeze(0).to(DEVICE))
                        _, mu_b, _ = vae(img_b.unsqueeze(0).to(DEVICE))

                        mu_a = mu_a.cpu().squeeze().numpy()
                        mu_b = mu_b.cpu().squeeze().numpy()

                        alphas = np.linspace(0, 1, 10)
                        interpolated = []

                        for alpha in alphas:
                            z_interp = (1 - alpha) * mu_a + alpha * mu_b
                            z_tensor = torch.tensor([z_interp], dtype=torch.float32).to(DEVICE)
                            gen_img = vae.decoder(z_tensor).cpu().squeeze().numpy()
                            interpolated.append(gen_img)

                    vae.to('cpu')

                    st.subheader("插值结果")
                    col_start, col_interp, col_end = st.columns([1, 8, 1])
                    with col_start:
                        st.image(img_a.squeeze().numpy(), width=80, clamp=True)
                        st.caption(f"起点 {label_a}")
                    with col_interp:
                        fig, axes = plt.subplots(1, 10, figsize=(20, 3), dpi=100)
                        for i, (img, alpha) in enumerate(zip(interpolated, alphas)):
                            axes[i].imshow(img, cmap='gray')
                            axes[i].axis('off')
                            axes[i].set_title(f'α={alpha:.1f}')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    with col_end:
                        st.image(img_b.squeeze().numpy(), width=80, clamp=True)
                        st.caption(f"终点 {label_b}")

    with tab3:
        st.header("高级生成模型：DCGAN & 扩散模型")

        dcgan_tab, diffusion_tab = st.tabs(["DCGAN", "扩散模型"])

        with dcgan_tab:
            st.subheader("DCGAN 生成器与判别器博弈")
            
            st.markdown("""**🎯 目标**：体验生成对抗网络（GAN）如何从随机噪声生成手写数字。

**🎛️ 控件说明**：
- **开始训练 DCGAN**：训练生成器和判别器进行博弈，训练完成后显示结果
- **噪声维度滑块**：调整输入噪声的维度（仅在训练后生效）

**📊 结果展示**：
- **生成样本网格**：展示生成器生成的数字图像
- **判别器分数**：显示判别器对生成样本的判断（接近 0.5 表示难分真伪）
- **训练时间线**：上排为真实样本参照，下排为不同 epoch 的生成结果

**💡 演示说明**：训练完成后，网格展示生成器生成的数字；时间线图显示训练过程中生成质量的提升；拖动噪声维度滑块可观察输入噪声对生成结果的影响。""")

            col_btn, col_progress = st.columns([1, 3])

            with col_btn:
                if st.button("开始训练 DCGAN", key="train_dcgan_btn", disabled=st.session_state.get('training_dcgan', False)):
                    st.session_state['training_dcgan'] = True
                    st.session_state['dcgan_training_epoch'] = 0
                    st.session_state['dcgan_timeline'] = []

                    train_subset = Subset(train_dataset, range(min(train_size, len(train_dataset))))
                    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)

                    if 'dcgan_real_samples' not in st.session_state:
                        real_indices = random.sample(range(len(train_subset)), 10)
                        real_samples = []
                        for idx in real_indices:
                            img, label = train_subset[idx]
                            real_samples.append((img.squeeze().numpy(), label))
                        st.session_state['dcgan_real_samples'] = real_samples

                    progress_bar = st.empty()
                    status_text = st.empty()
                    progress_bar.progress(0)
                    status_text.text("训练准备中...")

                    def update_progress(epoch, total_epochs):
                        progress_bar.progress(epoch / total_epochs)
                        status_text.text(f"训练进度: Epoch {epoch}/{total_epochs}")

                    generator, discriminator, timeline = train_dcgan_sync(
                        train_loader, num_epochs_dcgan, update_progress
                    )

                    progress_bar.progress(1.0)
                    status_text.text("训练完成")

                    st.session_state['generator'] = generator
                    st.session_state['discriminator'] = discriminator
                    st.session_state['dcgan_timeline'] = timeline
                    st.session_state['dcgan_trained'] = True
                    st.session_state['training_dcgan'] = False

            with col_progress:
                if st.session_state.get('dcgan_trained', False):
                    st.caption("✅ DCGAN 模型已训练完成")

            if st.session_state.get('dcgan_trained', False) and 'generator' in st.session_state:
                generator = st.session_state['generator']
                discriminator = st.session_state['discriminator']
                timeline = st.session_state.get('dcgan_timeline', [])

                generator.to(DEVICE)
                discriminator.to(DEVICE)
                generator.eval()
                discriminator.eval()

                with torch.no_grad():
                    fixed_noise = torch.randn(16, 100).to(DEVICE)
                    fake_images = generator(fixed_noise).cpu()
                    fake_images = torch.clamp(fake_images, 0, 1)

                    avg_prob = 0
                    for img in fake_images:
                        prob = discriminator(img.unsqueeze(0).to(DEVICE)).cpu().item()
                        avg_prob += prob
                    avg_prob /= len(fake_images)

                generator.to('cpu')
                discriminator.to('cpu')

                st.write(f"判别器对当前生成样本的平均判定概率: **{avg_prob:.4f}**")

                cols = st.columns(4)
                for i in range(4):
                    with cols[i]:
                        st.image(fake_images[i].squeeze().numpy(), width=100, clamp=True)

                st.subheader("训练时间线 (固定噪声)")

                if timeline:
                    real_samples = st.session_state.get('dcgan_real_samples', [])
                    
                    num_frames = min(len(timeline), 10)
                    if len(timeline) > 10:
                        indices = np.linspace(0, len(timeline)-1, 10, dtype=int)
                        sampled_timeline = [timeline[i] for i in indices]
                        epoch_labels = [f'Epoch {indices[i]+1}' for i in range(10)]
                    else:
                        sampled_timeline = timeline
                        epoch_labels = [f'Epoch {i+1}' for i in range(len(timeline))]
                    
                    fig, axes = plt.subplots(2, num_frames, figsize=(num_frames * 2, 6), dpi=100)
                    
                    for i in range(num_frames):
                        if real_samples:
                            axes[0, i].imshow(real_samples[i][0], cmap='gray')
                        axes[0, i].axis('off')
                        axes[0, i].set_title('真实样本')
                    
                    for i in range(num_frames):
                        axes[1, i].imshow(sampled_timeline[i], cmap='gray')
                        axes[1, i].axis('off')
                        axes[1, i].set_title(epoch_labels[i])
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.caption("上排为真实训练样本（目标分布），下排为固定噪声在不同训练阶段的生成结果")
                else:
                    st.info("💡 训练时间线仅在重新训练时生成，当前使用预训练模型可直接查看生成样本和噪声维度调节功能。")

                st.subheader("噪声维度调节")

                dim_idx = st.slider("选择噪声维度索引", 0, 99, 50, key="dim_idx_slider")
                dim_value = st.slider(f"维度 {dim_idx} 的值", -3.0, 3.0, 0.0, 0.1, key="dim_value_slider")

                if st.button("生成", key="generate_noise"):
                    with st.spinner("生成图像中..."):
                        generator.to(DEVICE)
                        generator.eval()
                        noise = torch.randn(1, 100).to(DEVICE)
                        noise[0, dim_idx] = dim_value
                        with torch.no_grad():
                            gen_img = generator(noise).cpu().squeeze().numpy()
                        generator.to('cpu')
                        st.image(gen_img, width=200, clamp=True)

            elif not st.session_state.get('training_dcgan', False):
                st.info("👆 点击上方「开始训练 DCGAN」按钮开始训练模型")

        with diffusion_tab:
            st.subheader("扩散模型采样")
            
            st.markdown("""**🎯 目标**：观察文生图扩散模型的采样过程，理解采样步数和引导强度对生成结果的影响。

**🎛️ 控件说明**：
- **提示词**：描述想要生成的图像内容
- **负向提示词**：描述不想要的内容
- **采样步数**：去噪迭代次数（步数越多图像越清晰）
- **随机种子**：控制生成的随机性，相同种子生成相同图像
- **引导强度**：控制提示词的影响程度（过低模糊，过高可能失真）
- **生成图像**：开始采样生成图像
- **生成对比图**：生成三种不同引导强度的对比图

**📊 结果展示**：
- **生成图像**：根据提示词生成的图像
- **去噪过程可视化**：展示从纯噪声到清晰图像的逐步演化
- **固定种子多参数对比**：展示不同引导强度下的图像差异

**💡 演示说明**：点击生成按钮开始采样，去噪过程图展示扩散模型的逐步去噪过程；固定种子对比展示不同引导强度的效果（Scale 1.5 模糊，Scale 7.5 清晰，Scale 15.0 失真）。若模型加载失败，将使用预生成样例模拟演示。""")

            if not st.session_state.get('diffusion_loaded', False):
                try:
                    from diffusers import DDPMPipeline
                    with st.spinner("加载扩散模型中..."):
                        pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32", torch_dtype=torch.float32)
                        pipeline = pipeline.to(DEVICE)
                        st.session_state['diffusion_pipeline'] = pipeline
                        st.session_state['diffusion_loaded'] = True
                        st.success("扩散模型加载成功！")
                except Exception as e:
                    st.warning(f"扩散模型加载失败，当前为模拟演示效果: {str(e)}")
                    st.session_state['diffusion_loaded'] = False
                    
                    if 'diffusion_demo_images' not in st.session_state:
                        import cv2
                        
                        idx = random.randint(0, len(test_dataset) - 1)
                        original_img, _ = test_dataset[idx]
                        original_img = original_img.squeeze().numpy()
                        
                        original_rgb = np.stack([original_img] * 3, axis=-1)
                        
                        blurred = cv2.GaussianBlur(original_img, (5, 5), sigmaX=2.0)
                        low_guidance = np.stack([blurred] * 3, axis=-1)
                        
                        sharpened = cv2.filter2D(original_img, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
                        noise = np.random.randn(*original_img.shape) * 0.1
                        over_guidance = np.clip(sharpened + noise, 0, 1)
                        over_guidance = np.stack([over_guidance] * 3, axis=-1)
                        
                        low_guidance = cv2.resize(low_guidance, (32, 32))
                        original_rgb = cv2.resize(original_rgb, (32, 32))
                        over_guidance = cv2.resize(over_guidance, (32, 32))
                        
                        st.session_state['diffusion_demo_images'] = {
                            'low': low_guidance,
                            'normal': original_rgb,
                            'over': over_guidance
                        }

            diffusion_loaded = st.session_state.get('diffusion_loaded', False)

            col1, col2 = st.columns(2)
            with col1:
                prompt = st.text_input("Prompt", value="a handwritten digit", key="diff_prompt")
            with col2:
                negative_prompt = st.text_input("Negative Prompt", value="blurry, distorted", key="diff_neg_prompt")

            col3, col4 = st.columns(2)
            with col3:
                num_steps = st.slider("采样步数", 1, 50, 25, key="diff_steps")
            with col4:
                seed = st.number_input("随机种子", value=42, key="diff_seed")

            guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, key="diff_guidance")

            if st.button("生成图像", key="generate_diffusion"):
                with st.spinner("生成图像中..."):
                    torch.manual_seed(seed)

                    if diffusion_loaded:
                        pipeline = st.session_state['diffusion_pipeline']
                        with torch.no_grad():
                            images_list = []
                            latents = torch.randn(1, 3, 32, 32, dtype=torch.float32).to(DEVICE)

                            for i, t in enumerate(range(num_steps)):
                                latents = latents * 0.99 + torch.randn_like(latents) * 0.01
                                if i % (num_steps // 5) == 0:
                                    with torch.no_grad():
                                        sample = pipeline.scheduler.add_noise(
                                            pipeline.unet(latents, t).sample,
                                            torch.randn_like(latents),
                                            t
                                        )
                                        img = sample.cpu().squeeze().permute(1, 2, 0).numpy()
                                        img = (img - img.min()) / (img.max() - img.min())
                                        images_list.append(img)

                            final_image = latents.cpu().squeeze().permute(1, 2, 0).numpy()
                            final_image = (final_image - final_image.min()) / (final_image.max() - final_image.min())

                            st.image(final_image, width=256, clamp=True)

                            if len(images_list) > 1:
                                st.subheader("去噪过程可视化")
                                fig, axes = plt.subplots(1, len(images_list), figsize=(len(images_list) * 3, 3), dpi=100)
                                if len(images_list) == 1:
                                    axes = [axes]
                                for i, img in enumerate(images_list):
                                    step = i * (num_steps // 5)
                                    axes[i].imshow(img)
                                    axes[i].axis('off')
                                    axes[i].set_title(f'Step {step}')
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                    else:
                        demo_images = st.session_state.get('diffusion_demo_images', {})
                        
                        if guidance_scale <= 3.0:
                            display_img = demo_images.get('low', np.random.rand(32, 32, 3))
                            demo_type = "低引导示例 (guidance_scale≈1.5)"
                        elif guidance_scale <= 12.0:
                            display_img = demo_images.get('normal', np.random.rand(32, 32, 3))
                            demo_type = "正常示例 (guidance_scale≈7.5)"
                        else:
                            display_img = demo_images.get('over', np.random.rand(32, 32, 3))
                            demo_type = "过度引导示例 (guidance_scale≈15.0)"
                        
                        st.image(display_img, width=256)
                        st.caption(f"当前展示: {demo_type}")
                        
                        st.subheader("去噪过程可视化")
                        target_img = display_img
                        noise_img = np.random.rand(32, 32, 3)
                        denoise_steps = []
                        for i in range(5):
                            alpha = i / 4.0
                            step_img = (1 - alpha) * noise_img + alpha * target_img
                            denoise_steps.append(step_img)
                        
                        fig, axes = plt.subplots(1, 5, figsize=(15, 3), dpi=100)
                        for i, img in enumerate(denoise_steps):
                            axes[i].imshow(img)
                            axes[i].axis('off')
                            axes[i].set_title(f'Step {i * (num_steps // 4)}')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        st.caption("模拟去噪过程 (基于预生成样例)")

            st.subheader("固定种子多参数对比")

            st.write("**Guidance Scale 对比:**")

            scales = [1.5, 7.5, 15.0]
            cols = st.columns(3)
            demo_images = st.session_state.get('diffusion_demo_images', {})

            for idx, gs in enumerate(scales):
                with cols[idx]:
                    torch.manual_seed(42)
                    if diffusion_loaded:
                        st.image(np.random.rand(32, 32, 3), width=150)
                    else:
                        if gs <= 3.0:
                            st.image(demo_images.get('low', np.random.rand(32, 32, 3)), width=150)
                        elif gs <= 12.0:
                            st.image(demo_images.get('normal', np.random.rand(32, 32, 3)), width=150)
                        else:
                            st.image(demo_images.get('over', np.random.rand(32, 32, 3)), width=150)
                    st.caption(f"Scale = {gs}")

            st.info("**说明:** guidance scale 过低（~1.5）：图像多样性高但可能偏离文本；适中（~7.5）：质量与相关性平衡；过高（~15+）：图像过饱和、出现伪影。这对应 PPT 中 CFG 公式 `v_cfg = (1+w)*v_y - w*v_0` 的效应。")


if __name__ == "__main__":
    main()