import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.font_manager as fm

_font_candidates = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
_available = [f.name for f in fm.fontManager.ttflist]
plt.rcParams['font.sans-serif'] = [f for f in _font_candidates if f in _available] or ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "saved_models"
MODEL_PATH.mkdir(exist_ok=True, parents=True)

@st.cache_resource
def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

class RotationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_features(self, x):
        x = self.conv(x)
        return x.reshape(x.size(0), -1)

class JigsawCNN(nn.Module):
    def __init__(self, num_pieces=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4 * num_pieces, 256),
            nn.ReLU(),
            nn.Linear(256, num_pieces * num_pieces)
        )
    
    def forward(self, x):
        batch_size, num_pieces, c, h, w = x.size()
        x = x.reshape(batch_size * num_pieces, c, h, w)
        x = self.conv(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x.reshape(batch_size, 4, 4)

    def get_features(self, x):
        batch_size, num_pieces, c, h, w = x.size()
        x = x.reshape(batch_size * num_pieces, c, h, w)
        x = self.conv(x)
        return x.reshape(batch_size, -1)

class MAEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask):
        masked_x = x * mask
        encoded = self.encoder(masked_x)
        decoded = self.decoder(encoded)
        return decoded

class SimCLREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

def generate_rotations(images):
    rotations = []
    labels = []
    for img in images:
        for angle in [0, 90, 180, 270]:
            rotated = F.rotate(img, angle)
            rotations.append(rotated)
            labels.append(angle // 90)
    return torch.stack(rotations), torch.tensor(labels)

def create_jigsaw_puzzle(image, piece_size=32//2):
    pieces = []
    positions = list(range(4))
    np.random.shuffle(positions)
    
    for i in range(2):
        for j in range(2):
            piece = image[:, i*piece_size:(i+1)*piece_size, j*piece_size:(j+1)*piece_size]
            pieces.append(piece)
    
    shuffled_pieces = [pieces[p] for p in positions]
    return torch.stack(shuffled_pieces), torch.tensor(positions)

def infonce_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    N = z.size(0)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(N, device=z.device) * 1e9
    sim = sim - mask
    labels = torch.arange(N, device=z.device)
    labels[:N//2] = labels[N//2:]
    labels[N//2:] = labels[:N//2]
    return nn.CrossEntropyLoss()(sim, labels)

def train_rotation_model(model, train_loader, epochs=2, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    accs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, _ in train_loader:
            images, labels = generate_rotations(images)
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        losses.append(avg_loss)
        accs.append(acc)
        
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH / "rotation_model.pth")
    return losses, accs

def train_jigsaw_model(model, train_loader, epochs=2, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    accs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, _ in train_loader:
            batch_puzzles = []
            batch_labels = []
            for img in images:
                puzzle, pos = create_jigsaw_puzzle(img)
                batch_puzzles.append(puzzle)
                batch_labels.append(pos)
            
            puzzles = torch.stack(batch_puzzles).to(device)
            labels = torch.stack(batch_labels).to(device)
            
            optimizer.zero_grad()
            outputs = model(puzzles)
            loss = criterion(outputs.reshape(-1, 4), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(2)
            total += labels.numel()
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        losses.append(avg_loss)
        accs.append(acc)
        
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH / "jigsaw_model.pth")
    return losses, accs

def train_mae_model(model, train_loader, mask_ratio=0.75, epochs=2, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, _ in train_loader:
            images = images.to(device)
            mask = torch.rand(images.size(0), 1, images.size(2), images.size(3), device=device) > mask_ratio
            optimizer.zero_grad()
            outputs = model(images, mask)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.6f}")
    
    torch.save(model.state_dict(), MODEL_PATH / "mae_model.pth")
    return losses

def train_simclr_model(model, train_loader, epochs=3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    
    augment1 = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ])
    augment2 = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3)
    ])
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, _ in train_loader:
            images = images.to(device)
            z1 = model(augment1(images))
            z2 = model(augment2(images))
            loss = infonce_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH / "simclr_model.pth")
    return losses

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    st.pyplot(plt)
    plt.close()

def plot_tsne(features, labels, title='t-SNE 特征分布'):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.legend(handles=scatter.legend_elements()[0], labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], title='类别')
    plt.title(title)
    st.pyplot(plt)
    plt.close()

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b).item()

def main():
    st.set_page_config(layout="wide", page_title="自监督学习交互式演示")
    st.title("自监督学习交互式演示")
    
    train_dataset, test_dataset = load_cifar10()
    device = 'cpu'
    class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    
    with st.sidebar:
        st.header("全局设置")
        use_pretrained = st.checkbox("使用已训练好的模型", value=True)
        
        if use_pretrained:
            st.success("✅ 使用已训练好的模型，无需重新训练！")
        else:
            quick_mode = st.checkbox("快速模式", value=True)
            
            if quick_mode:
                st.info("快速模式：训练样本 500，epoch=5")
                train_samples = 500
                epochs = 5
            else:
                train_samples = st.slider("训练样本数", 1000, 10000, 5000, step=1000)
                epochs = st.slider("训练轮数", 1, 10, 5)
        
        st.divider()
        st.header("功能选择")
        function_choice = st.selectbox(
            "选择功能",
            ["旋转预测", "拼图重排", "MAE 遮挡重建", "SimCLR 对比学习", "实时演示", "对比结果汇总"]
        )
    
    st.session_state['use_pretrained'] = use_pretrained
    if not use_pretrained:
        st.session_state['quick_mode'] = quick_mode
        st.session_state['train_samples'] = train_samples
        st.session_state['epochs'] = epochs
    
    if function_choice == "旋转预测":
        st.header("🔄 旋转预测")
        
        st.markdown("""
        ### 📖 功能说明
        
        **任务目标**：让模型学会判断图片旋转了多少度（0°、90°、180°或270°）。这是一种自监督学习任务，模型通过学习识别旋转角度来提取有用的图像特征。
        
        **页面布局**：
        - **训练区**：点击「开始训练」按钮训练模型（仅当不使用预训练模型时显示）
        - **演示区**：可以上传图片或随机选择测试图片，点击「随机旋转并预测」按钮
        
        **结果解释**：
        - **预测演示**：展示原图、旋转后的图、模型预测结果及置信度
        """)
        
        if use_pretrained:
            # 检查模型是否存在
            if (MODEL_PATH / "rotation_model.pth").exists():
                st.success("✅ 已加载旋转预测预训练模型！")
                # 直接显示演示区域，使用更宽的布局
                col_demo = st.container()
            else:
                st.warning("⚠️ 未找到预训练模型，将使用随机初始化的模型")
                col_demo = st.container()
        else:
            col_train, col_demo = st.columns(2)
        
        if not use_pretrained:
            with col_train:
                st.subheader("📚 模型训练")
                if st.button("开始训练"):
                    ts = st.session_state['train_samples']
                    ep = st.session_state['epochs']
                    subset = Subset(train_dataset, np.random.choice(len(train_dataset), ts, replace=False))
                    train_loader = DataLoader(subset, batch_size=32, shuffle=True)
                    model = RotationCNN()
                    losses, accs = train_rotation_model(model, train_loader, epochs=ep, device=device)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.plot(losses, label='训练损失')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    ax2.plot(accs, label='准确率')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Accuracy')
                    ax2.legend()
                    st.pyplot(fig)
                    plt.close()
                    
                    model.eval()
                    test_features = []
                    test_labels = []
                    pred_list = []
                    for images, labels in DataLoader(test_dataset, batch_size=32):
                        images, rot_labels = generate_rotations(images)
                        with torch.no_grad():
                            outputs = model(images.to(device))
                        _, predicted = outputs.max(1)
                        test_features.extend(model.get_features(images.to(device)).detach().cpu().numpy())
                        test_labels.extend(rot_labels.numpy())
                        pred_list.extend(predicted.cpu().numpy())
                    
                    st.subheader("📊 训练结果")
                    plot_confusion_matrix(test_labels[:1000], pred_list[:1000], ['0°', '90°', '180°', '270°'])
                    plot_tsne(np.array(test_features[:500]), np.array(test_labels[:500]))
        
        with col_demo:
            st.subheader("🎯 演示")
            uploaded_file = st.file_uploader("上传图片（32x32像素效果最佳）", type=["png", "jpg"])
            
            if not uploaded_file:
                if st.button("随机选择一张测试图片"):
                    st.session_state['rot_random_idx'] = np.random.randint(0, len(test_dataset))
                idx = st.session_state.get('rot_random_idx', 0)
                img_np = test_dataset[idx][0].permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
                img = Image.fromarray((img_np * 255).astype(np.uint8))
                st.write(f"当前图片类别: {class_names[test_dataset[idx][1]]}")
            else:
                img = Image.open(uploaded_file).resize((32, 32))
            
            st.image(img, caption='原图', width=150)
            
            if st.button("随机旋转并预测"):
                angles = [0, 90, 180, 270]
                true_angle_idx = np.random.randint(0, 4)
                true_angle = angles[true_angle_idx]
                
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                img_tensor = transform(img).unsqueeze(0)
                
                rotated_tensor = F.rotate(img_tensor[0], true_angle)
                
                model = RotationCNN()
                if (MODEL_PATH / "rotation_model.pth").exists():
                    model.load_state_dict(torch.load(MODEL_PATH / "rotation_model.pth", map_location=device))
                model.to(device)
                model.eval()
                
                with torch.no_grad():
                    output = model(rotated_tensor.unsqueeze(0).to(device))
                    probs = torch.softmax(output, dim=1)
                    pred_angle_idx = torch.argmax(probs[0]).item()
                    pred_angle = angles[pred_angle_idx]
                    confidence = probs[0][pred_angle_idx].item()
                
                corrected_img = F.rotate(img, -pred_angle)
                
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(img)
                axes[0].set_title('原图')
                axes[0].axis('off')
                
                rot_np = rotated_tensor.permute(1, 2, 0).cpu().numpy()
                rot_np = (rot_np - rot_np.min()) / (rot_np.max() - rot_np.min() + 1e-6)
                axes[1].imshow(rot_np)
                axes[1].set_title(f'随机旋转后\n真实角度: {true_angle}°')
                axes[1].axis('off')
                
                axes[2].imshow(corrected_img)
                axes[2].set_title(f'模型预测\n预测角度: {pred_angle}°\n置信度: {confidence:.4f}')
                axes[2].axis('off')
                
                if pred_angle == true_angle:
                    axes[3].text(0.5, 0.5, '✓ 预测正确!', fontsize=16, ha='center', va='center', color='green')
                else:
                    axes[3].text(0.5, 0.5, '✗ 预测错误', fontsize=16, ha='center', va='center', color='red')
                axes[3].axis('off')
                
                st.pyplot(fig)
                plt.close()
                
                if pred_angle == true_angle:
                    st.success(f"✅ 预测正确! 真实角度 {true_angle}°, 预测角度 {pred_angle}°")
                else:
                    st.error(f"❌ 预测错误! 真实角度 {true_angle}°, 预测角度 {pred_angle}°")
    
    elif function_choice == "拼图重排":
        st.header("🧩 拼图重排")
        
        st.markdown("""
        ### 📖 功能说明
        
        **任务目标**：让模型学会将打乱的拼图块重新排列到正确位置。图片会被分成2×2的4个块，打乱顺序后让模型预测每块应该放在哪个位置。
        
        **页面布局**：
        - **训练区**：点击「开始训练」按钮训练模型（仅当不使用预训练模型时显示）
        - **演示区**：可以上传图片或随机选择测试图片，点击「开始拼图重排」按钮查看模型的重建效果。
        
        **结果解释**：
        - **原图**：原始的完整图片
        - **打乱后**：图片被切成4块并打乱顺序后的样子
        - **模型重建**：模型根据打乱的块预测每个块的正确位置后重建的图片
        """)
        
        if use_pretrained:
            if (MODEL_PATH / "jigsaw_model.pth").exists():
                st.success("✅ 已加载拼图重排预训练模型！")
                col_demo = st.container()
            else:
                st.warning("⚠️ 未找到预训练模型，将使用随机初始化的模型")
                col_demo = st.container()
        else:
            col_train, col_demo = st.columns(2)
        
        if not use_pretrained:
            with col_train:
                st.subheader("📚 模型训练")
                if st.button("开始训练"):
                    ts = st.session_state['train_samples']
                    ep = st.session_state['epochs']
                    subset = Subset(train_dataset, np.random.choice(len(train_dataset), ts, replace=False))
                    train_loader = DataLoader(subset, batch_size=16, shuffle=True)
                    model = JigsawCNN()
                    losses, accs = train_jigsaw_model(model, train_loader, epochs=ep, device=device)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.plot(losses, label='训练损失')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    ax2.plot(accs, label='准确率')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Accuracy')
                    ax2.legend()
                    st.pyplot(fig)
                    plt.close()
        
        with col_demo:
            st.subheader("🎯 演示")
            uploaded_file = st.file_uploader("上传图片（32x32像素效果最佳）", type=["png", "jpg"])
            
            if not uploaded_file:
                if st.button("随机选择一张测试图片"):
                    st.session_state['jig_random_idx'] = np.random.randint(0, len(test_dataset))
                idx = st.session_state.get('jig_random_idx', 0)
                img_np = test_dataset[idx][0].permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
                img = Image.fromarray((img_np * 255).astype(np.uint8))
                st.write(f"当前图片类别: {class_names[test_dataset[idx][1]]}")
            else:
                img = Image.open(uploaded_file).resize((32, 32))
            
            st.image(img, caption='待处理图片', width=150)
            
            if st.button("开始拼图重排"):
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                img_tensor = transform(img)
                
                puzzle, true_positions = create_jigsaw_puzzle(img_tensor)
                puzzle_input = puzzle.unsqueeze(0)
                
                model = JigsawCNN()
                if (MODEL_PATH / "jigsaw_model.pth").exists():
                    model.load_state_dict(torch.load(MODEL_PATH / "jigsaw_model.pth", map_location=device))
                model.to(device)
                model.eval()
                
                with torch.no_grad():
                    output = model(puzzle_input.to(device))
                    pred_positions = torch.argmax(output[0], dim=1)
                
                piece_size = 32 // 2
                reconstructed = torch.zeros(3, 32, 32)
                for i in range(4):
                    row = pred_positions[i] // 2
                    col = pred_positions[i] % 2
                    reconstructed[:, row*piece_size:(row+1)*piece_size, col*piece_size:(col+1)*piece_size] = puzzle[i]
                
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img)
                axes[0].set_title('原图')
                axes[0].axis('off')
                
                shuffled_img = torch.zeros(3, 32, 32)
                for i, pos in enumerate(true_positions):
                    row = i // 2
                    col = i % 2
                    shuffled_img[:, row*piece_size:(row+1)*piece_size, col*piece_size:(col+1)*piece_size] = puzzle[pos]
                shuffled_np = shuffled_img.permute(1, 2, 0).cpu().numpy()
                shuffled_np = (shuffled_np - shuffled_np.min()) / (shuffled_np.max() - shuffled_np.min() + 1e-6)
                axes[1].imshow(shuffled_np)
                axes[1].set_title('打乱后 (2x2)')
                axes[1].axis('off')
                
                recon_np = reconstructed.permute(1, 2, 0).cpu().numpy()
                recon_np = (recon_np - recon_np.min()) / (recon_np.max() - recon_np.min() + 1e-6)
                axes[2].imshow(recon_np)
                axes[2].set_title('模型重建')
                axes[2].axis('off')
                
                st.pyplot(fig)
                plt.close()
    
    elif function_choice == "MAE 遮挡重建":
        st.header("🎨 MAE 遮挡重建")
        
        st.markdown("""
        ### 📖 功能说明
        
        **任务目标**：让模型学会根据图片未被遮挡的部分，重建出被遮挡的区域。这就像让AI成为"拼图高手"，只看图片的一部分就能想象出完整的画面。
        
        **页面布局**：
        - **训练区**：点击「开始训练」按钮训练模型（仅当不使用预训练模型时显示）
        - **演示区**：可以上传图片或随机选择测试图片，调整遮挡比例后点击「开始遮挡重建」按钮查看效果。
        
        **结果解释**：
        - **原图**：原始的完整图片
        - **遮挡后**：随机遮挡部分区域后的图片
        - **重建图**：模型根据可见部分重建的完整图片
        - **误差图**：显示重建结果与原图的差异（越亮表示差异越大）
        - **MSE值**：重建误差的量化指标，数值越小表示重建质量越好
        """)
        
        if use_pretrained:
            if (MODEL_PATH / "mae_model.pth").exists():
                st.success("✅ 已加载MAE遮挡重建预训练模型！")
                col_demo = st.container()
            else:
                st.warning("⚠️ 未找到预训练模型，将使用随机初始化的模型")
                col_demo = st.container()
        else:
            col_train, col_demo = st.columns(2)
        
        if not use_pretrained:
            with col_train:
                st.subheader("📚 模型训练")
                if st.button("开始训练"):
                    ts = st.session_state['train_samples']
                    ep = st.session_state['epochs']
                    subset = Subset(train_dataset, np.random.choice(len(train_dataset), ts, replace=False))
                    train_loader = DataLoader(subset, batch_size=32, shuffle=True)
                    model = MAEModel()
                    losses = train_mae_model(model, train_loader, epochs=ep, device=device)
                    
                    fig = plt.figure(figsize=(8, 4))
                    plt.plot(losses, label='训练损失')
                    plt.xlabel('Epoch')
                    plt.ylabel('MSE Loss')
                    plt.legend()
                    st.pyplot(fig)
                    plt.close()
        
        with col_demo:
            st.subheader("🎯 演示")
            uploaded_file = st.file_uploader("上传图片（32x32像素效果最佳）", type=["png", "jpg"])
            mask_ratio = st.slider("选择遮挡比例", 0.1, 0.9, 0.75, step=0.05)
            
            if not uploaded_file:
                if st.button("随机选择一张测试图片"):
                    st.session_state['mae_random_idx'] = np.random.randint(0, len(test_dataset))
                idx = st.session_state.get('mae_random_idx', 0)
                img_np = test_dataset[idx][0].permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
                img = Image.fromarray((img_np * 255).astype(np.uint8))
                st.write(f"当前图片类别: {class_names[test_dataset[idx][1]]}")
            else:
                img = Image.open(uploaded_file).resize((32, 32))
            
            st.image(img, caption='待处理图片', width=150)
            
            if st.button("开始遮挡重建"):
                transform = transforms.Compose([transforms.ToTensor()])
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                model = MAEModel()
                if (MODEL_PATH / "mae_model.pth").exists():
                    model.load_state_dict(torch.load(MODEL_PATH / "mae_model.pth", map_location=device))
                model.to(device)
                model.eval()
                
                mask = torch.rand(img_tensor.size(0), 1, img_tensor.size(2), img_tensor.size(3), device=device) > mask_ratio
                with torch.no_grad():
                    output = model(img_tensor, mask)
                
                mse_loss = nn.MSELoss()(output * (~mask), img_tensor * (~mask)).item()
                
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                
                axes[0].imshow(img)
                axes[0].set_title('原图')
                axes[0].axis('off')
                
                masked_img = img_tensor * mask
                masked_np = masked_img[0].permute(1, 2, 0).cpu().numpy()
                axes[1].imshow(masked_np)
                axes[1].set_title(f'遮挡后（{int(mask_ratio*100)}%被遮挡）')
                axes[1].axis('off')
                
                recon_np = output[0].permute(1, 2, 0).cpu().numpy()
                axes[2].imshow(recon_np)
                axes[2].set_title('重建图')
                axes[2].axis('off')
                
                diff = np.abs(recon_np - masked_np)
                axes[3].imshow(diff, cmap='gray')
                axes[3].set_title(f'误差图 (MSE={mse_loss:.4f})')
                axes[3].axis('off')
                
                st.pyplot(fig)
                plt.close()
            
            st.divider()
            if st.button("不同遮挡比例对比"):
                ratios = [0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85]
                mse_values = []
                
                transform = transforms.Compose([transforms.ToTensor()])
                img_tensor = test_dataset[0][0].unsqueeze(0).to(device)
                
                model = MAEModel()
                if (MODEL_PATH / "mae_model.pth").exists():
                    model.load_state_dict(torch.load(MODEL_PATH / "mae_model.pth", map_location=device))
                model.to(device)
                model.eval()
                
                for ratio in ratios:
                    mask = torch.rand(img_tensor.size(0), 1, img_tensor.size(2), img_tensor.size(3), device=device) > ratio
                    with torch.no_grad():
                        output = model(img_tensor, mask)
                    mse = nn.MSELoss()(output * (~mask), img_tensor * (~mask)).item()
                    mse_values.append(mse)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                ax1.bar([str(r) for r in ratios], mse_values)
                ax1.set_xlabel('遮挡比例')
                ax1.set_ylabel('MSE')
                ax1.set_title('遮挡比例 vs 重建误差')
                
                ax2.plot(ratios, mse_values, marker='o')
                ax2.set_xlabel('遮挡比例')
                ax2.set_ylabel('MSE')
                ax2.set_title('遮挡比例 vs 重建误差曲线')
                
                st.pyplot(fig)
                plt.close()
                
                df = pd.DataFrame({'遮挡比例': ratios, '重建MSE': mse_values})
                st.dataframe(df)
    
    elif function_choice == "SimCLR 对比学习":
        st.header("🔗 SimCLR 对比学习")
        
        st.markdown("""
        ### 📖 功能说明

        **任务目标**：让模型学习区分同一图片的不同增强版本（正样本）和不同图片（负样本），从而提取有意义的特征表示。

        **核心概念**：
        - **正样本对**: 同一张图片的两种不同增强。理想情况下它们的特征应该非常相似（余弦相似度高）。
        - **负样本对**: 两张不同的图片。理想情况下它们的特征应该不相似（余弦相似度低）。
        - **余弦相似度**: 范围 [-1, 1]，值越接近 1 表示两个特征越相似。
        - **训练好 vs 随机编码器**: 训练好的编码器能让正样本对更相似，负样本对更不相似。

        **页面布局**：
        - **训练区**：点击「开始训练」按钮训练对比学习模型（仅当不使用预训练模型时显示）
        - **演示区**：选择图片和增强方式，点击「计算相似度」查看结果。
        """)
        
        if use_pretrained:
            if (MODEL_PATH / "simclr_model.pth").exists():
                st.success("✅ 已加载SimCLR对比学习预训练模型！")
                col_demo = st.container()
            else:
                st.warning("⚠️ 未找到预训练模型，将使用随机初始化的模型")
                col_demo = st.container()
        else:
            col_train, col_demo = st.columns(2)
        
        if not use_pretrained:
            with col_train:
                st.subheader("📚 模型训练")
                if st.button("开始训练"):
                    ts = st.session_state['train_samples']
                    ep = st.session_state['epochs']
                    subset = Subset(train_dataset, np.random.choice(len(train_dataset), ts, replace=False))
                    train_loader = DataLoader(subset, batch_size=32, shuffle=True)
                    model = SimCLREncoder()
                    losses = train_simclr_model(model, train_loader, epochs=ep, device=device)
                    
                    fig = plt.figure(figsize=(8, 4))
                    plt.plot(losses, label='训练损失')
                    plt.xlabel('Epoch')
                    plt.ylabel('InfoNCE Loss')
                    plt.legend()
                    st.pyplot(fig)
                    plt.close()
        
        with col_demo:
            st.subheader("🎯 相似度对比")
            uploaded_file = st.file_uploader("上传图片（32x32像素效果最佳）", type=["png", "jpg"])
            
            if not uploaded_file:
                if st.button("随机选择一张测试图片"):
                    st.session_state['simclr_random_idx'] = np.random.randint(0, len(test_dataset))
                idx = st.session_state.get('simclr_random_idx', 0)
                img_np = test_dataset[idx][0].permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
                img = Image.fromarray((img_np * 255).astype(np.uint8))
                st.write(f"当前图片类别: {class_names[test_dataset[idx][1]]}")
            else:
                img = Image.open(uploaded_file).resize((32, 32))
            
            st.image(img, caption='待处理图片', width=150)
            
            aug_options = ["随机裁剪+颜色抖动", "随机旋转+高斯模糊", "灰度+水平翻转"]
            aug1 = st.selectbox("选择增强方式1", aug_options, index=0)
            aug2 = st.selectbox("选择增强方式2", aug_options, index=1)
            
            encoder_type = st.radio("选择编码器", ["训练好的编码器", "随机编码器"])
            
            if st.button("计算相似度"):
                augment_dict = {
                    "随机裁剪+颜色抖动": transforms.Compose([
                        transforms.RandomResizedCrop(32),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                    ]),
                    "随机旋转+高斯模糊": transforms.Compose([
                        transforms.RandomRotation(15),
                        transforms.GaussianBlur(kernel_size=3)
                    ]),
                    "灰度+水平翻转": transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.RandomHorizontalFlip()
                    ])
                }
                
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                model = SimCLREncoder()
                if encoder_type == "训练好的编码器" and (MODEL_PATH / "simclr_model.pth").exists():
                    model.load_state_dict(torch.load(MODEL_PATH / "simclr_model.pth", map_location=device))
                model.to(device)
                model.eval()
                
                aug_fn1 = augment_dict[aug1]
                aug_fn2 = augment_dict[aug2]
                
                with torch.no_grad():
                    z1 = model(aug_fn1(img_tensor))
                    z2 = model(aug_fn2(img_tensor))
                
                pos_sim = cosine_similarity(z1, z2)
                
                neg_idx = np.random.choice(len(test_dataset), 1)[0]
                neg_img = test_dataset[neg_idx][0].unsqueeze(0).to(device)
                with torch.no_grad():
                    z_neg = model(neg_img)
                neg_sim = cosine_similarity(z1, z_neg)
                
                st.write(f"正样本相似度（同一图片的两种增强）: **{pos_sim:.4f}**")
                st.write(f"负样本相似度（不同图片）: **{neg_sim:.4f}**")
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(['正样本对', '负样本对'], [pos_sim, neg_sim], color=['green', 'red'])
                ax.set_ylabel('余弦相似度')
                ax.set_title('正负样本相似度对比')
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                plt.close()
                
                all_combinations = []
                labels = []
                for i, aug_i in enumerate(aug_options):
                    for j, aug_j in enumerate(aug_options):
                        if i <= j:
                            aug_fn_i = augment_dict[aug_i]
                            aug_fn_j = augment_dict[aug_j]
                            with torch.no_grad():
                                zi = model(aug_fn_i(img_tensor))
                                zj = model(aug_fn_j(img_tensor))
                            sim = cosine_similarity(zi, zj)
                            all_combinations.append(sim)
                            labels.append(f"{aug_i}\nvs\n{aug_j}")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(labels, all_combinations)
                ax.set_ylabel('余弦相似度')
                ax.set_title('不同增强组合的相似度对比')
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                plt.close()
    
    elif function_choice == "实时演示":
        st.header("🎮 实时演示")
        
        st.markdown("""
        ### 📖 功能说明
        
        **任务目标**：综合演示所有自监督学习模型的效果。您可以选择任意一个已训练的模型，上传图片或选择测试图片，快速查看模型的推理结果。
        
        **使用方法**：
        1. 从下拉菜单选择要演示的模型（旋转预测、拼图重排或MAE）
        2. 上传一张图片或点击「随机选择一张测试图片」
        3. 点击「运行演示」按钮查看效果
        
        **演示效果**：
        - **旋转预测**：随机旋转图片后让模型预测旋转角度并校正
        - **拼图重排**：将图片切成4块打乱后让模型重建
        - **MAE**：随机遮挡部分区域后让模型重建完整图片
        """)
        
        model_type = st.selectbox("选择要演示的模型", ["旋转预测", "拼图重排", "MAE"])
        
        uploaded_file = st.file_uploader("上传图片（32x32像素效果最佳）", type=["png", "jpg"])
        
        if not uploaded_file:
            if st.button("随机选择一张测试图片"):
                st.session_state['realtime_random_idx'] = np.random.randint(0, len(test_dataset))
            idx = st.session_state.get('realtime_random_idx', 0)
            img_np = test_dataset[idx][0].permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
            img = Image.fromarray((img_np * 255).astype(np.uint8))
            st.write(f"当前图片类别: {class_names[test_dataset[idx][1]]}")
        else:
            img = Image.open(uploaded_file).resize((32, 32))
        
        st.image(img, caption='输入图片', width=150)
        
        if st.button("运行演示"):
            if model_type == "旋转预测":
                angles = [0, 90, 180, 270]
                true_angle_idx = np.random.randint(0, 4)
                true_angle = angles[true_angle_idx]
                
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                img_tensor = transform(img).unsqueeze(0)
                
                rotated_tensor = F.rotate(img_tensor[0], true_angle)
                
                model = RotationCNN()
                if (MODEL_PATH / "rotation_model.pth").exists():
                    model.load_state_dict(torch.load(MODEL_PATH / "rotation_model.pth", map_location=device))
                model.to(device)
                model.eval()
                
                with torch.no_grad():
                    output = model(rotated_tensor.unsqueeze(0).to(device))
                    probs = torch.softmax(output, dim=1)
                    pred_angle_idx = torch.argmax(probs[0]).item()
                    pred_angle = angles[pred_angle_idx]
                    confidence = probs[0][pred_angle_idx].item()
                
                corrected_img = F.rotate(img, -pred_angle)
                
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(img)
                axes[0].set_title('原图')
                axes[0].axis('off')
                
                rot_np = rotated_tensor.permute(1, 2, 0).cpu().numpy()
                rot_np = (rot_np - rot_np.min()) / (rot_np.max() - rot_np.min() + 1e-6)
                axes[1].imshow(rot_np)
                axes[1].set_title(f'随机旋转后\n真实角度: {true_angle}°')
                axes[1].axis('off')
                
                axes[2].imshow(corrected_img)
                axes[2].set_title(f'模型预测\n预测角度: {pred_angle}°\n置信度: {confidence:.4f}')
                axes[2].axis('off')
                
                if pred_angle == true_angle:
                    axes[3].text(0.5, 0.5, '✓ 预测正确!', fontsize=16, ha='center', va='center', color='green')
                else:
                    axes[3].text(0.5, 0.5, '✗ 预测错误', fontsize=16, ha='center', va='center', color='red')
                axes[3].axis('off')
                
                st.pyplot(fig)
                plt.close()
                
                if pred_angle == true_angle:
                    st.success(f"✅ 预测正确!")
                else:
                    st.error(f"❌ 预测错误!")
            
            elif model_type == "拼图重排":
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                img_tensor = transform(img)
                
                puzzle, true_positions = create_jigsaw_puzzle(img_tensor)
                puzzle_input = puzzle.unsqueeze(0)
                
                model = JigsawCNN()
                if (MODEL_PATH / "jigsaw_model.pth").exists():
                    model.load_state_dict(torch.load(MODEL_PATH / "jigsaw_model.pth", map_location=device))
                model.to(device)
                model.eval()
                
                with torch.no_grad():
                    output = model(puzzle_input.to(device))
                    pred_positions = torch.argmax(output[0], dim=1)
                
                piece_size = 32 // 2
                reconstructed = torch.zeros(3, 32, 32)
                for i in range(4):
                    row = pred_positions[i] // 2
                    col = pred_positions[i] % 2
                    reconstructed[:, row*piece_size:(row+1)*piece_size, col*piece_size:(col+1)*piece_size] = puzzle[i]
                
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img)
                axes[0].set_title('原图')
                axes[0].axis('off')
                
                shuffled_img = torch.zeros(3, 32, 32)
                for i, pos in enumerate(true_positions):
                    row = i // 2
                    col = i % 2
                    shuffled_img[:, row*piece_size:(row+1)*piece_size, col*piece_size:(col+1)*piece_size] = puzzle[pos]
                shuffled_np = shuffled_img.permute(1, 2, 0).cpu().numpy()
                shuffled_np = (shuffled_np - shuffled_np.min()) / (shuffled_np.max() - shuffled_np.min() + 1e-6)
                axes[1].imshow(shuffled_np)
                axes[1].set_title('打乱后 (2x2)')
                axes[1].axis('off')
                
                recon_np = reconstructed.permute(1, 2, 0).cpu().numpy()
                recon_np = (recon_np - recon_np.min()) / (recon_np.max() - recon_np.min() + 1e-6)
                axes[2].imshow(recon_np)
                axes[2].set_title('重建')
                axes[2].axis('off')
                
                st.pyplot(fig)
                plt.close()
            
            elif model_type == "MAE":
                transform = transforms.Compose([transforms.ToTensor()])
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                model = MAEModel()
                if (MODEL_PATH / "mae_model.pth").exists():
                    model.load_state_dict(torch.load(MODEL_PATH / "mae_model.pth", map_location=device))
                model.to(device)
                model.eval()
                
                mask = torch.rand(img_tensor.size(0), 1, img_tensor.size(2), img_tensor.size(3), device=device) > 0.75
                with torch.no_grad():
                    output = model(img_tensor, mask)
                
                mse_loss = nn.MSELoss()(output * (~mask), img_tensor * (~mask)).item()
                
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img)
                axes[0].set_title('原图')
                axes[0].axis('off')
                
                masked_img = img_tensor * mask
                masked_np = masked_img[0].permute(1, 2, 0).cpu().numpy()
                axes[1].imshow(masked_np)
                axes[1].set_title('遮挡后')
                axes[1].axis('off')
                
                recon_np = output[0].permute(1, 2, 0).cpu().numpy()
                axes[2].imshow(recon_np)
                axes[2].set_title(f'重建 (MSE={mse_loss:.4f})')
                axes[2].axis('off')
                
                st.pyplot(fig)
                plt.close()
    
    elif function_choice == "对比结果汇总":
        st.header("📊 对比结果汇总")
        
        st.markdown("""
        ### 📖 功能说明
        
        **任务目标**：汇总展示所有自监督学习任务的对比结果，帮助您直观理解不同模型的效果和特点。
        
        **三个对比维度**：
        1. **不同遮挡比例对比**：查看MAE模型在不同遮挡比例下的重建效果
        2. **不同数据增强方式对比**：查看SimCLR模型对不同增强组合的相似度感知能力
        3. **训练前后效果对比**：对比旋转预测模型训练前后的特征分布变化
        
        **如何使用**：点击每个选项卡中的「生成对比」按钮即可查看相应的对比结果。
        """)
        
        tab1, tab2, tab3 = st.tabs([
            "不同遮挡比例对比", 
            "不同数据增强方式对比", 
            "训练前后效果对比"
        ])
        
        with tab1:
            st.subheader("不同遮挡比例对比")
            st.markdown("""
            ### 📝 功能说明

            测试MAE模型在不同遮挡比例下的重建性能。遮挡比例越高，模型需要重建的部分越多，难度越大。

            **图表含义**：
            - X轴：遮挡比例（0.25表示25%的区域被遮挡）
            - Y轴：重建误差（MSE值）
            - 趋势：遮挡比例越高，重建误差通常越大

            **如何解读**：一个好的MAE模型应该在高遮挡比例下仍然保持较低的重建误差。
            """)
            if st.button("生成遮挡比例对比"):
                ratios = [0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85]
                mse_values = []
                
                transform = transforms.Compose([transforms.ToTensor()])
                img_tensor = test_dataset[0][0].unsqueeze(0).to(device)
                
                model = MAEModel()
                if (MODEL_PATH / "mae_model.pth").exists():
                    model.load_state_dict(torch.load(MODEL_PATH / "mae_model.pth", map_location=device))
                model.to(device)
                model.eval()
                
                for ratio in ratios:
                    mask = torch.rand(img_tensor.size(0), 1, img_tensor.size(2), img_tensor.size(3), device=device) > ratio
                    with torch.no_grad():
                        output = model(img_tensor, mask)
                    mse = nn.MSELoss()(output * (~mask), img_tensor * (~mask)).item()
                    mse_values.append(mse)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(ratios, mse_values, marker='o', label='MSE')
                ax.set_xlabel('遮挡比例')
                ax.set_ylabel('重建MSE')
                ax.set_title('遮挡比例 vs 重建误差')
                ax.legend()
                st.pyplot(fig)
                plt.close()
        
        with tab2:
            st.subheader("不同数据增强方式对比")
            st.markdown("""
            ### 📝 功能说明

            测试SimCLR模型对不同数据增强组合的相似度感知能力。

            **图表含义**：
            - X轴：不同的增强方式组合
            - Y轴：余弦相似度（值越高表示两种增强后的图片特征越相似）

            **如何解读**：
            - 相似度高：模型认为这两种增强方式产生的是同一张图片的不同版本
            - 相似度低：模型认为这两种增强方式产生的图片差异很大

            一个好的对比学习模型应该能够识别出同一张图片的不同增强版本。
            """)
            if st.button("生成增强方式对比"):
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                img_tensor = test_dataset[0][0].unsqueeze(0).to(device)
                
                aug_options = ["随机裁剪+颜色抖动", "随机旋转+高斯模糊", "灰度+水平翻转"]
                augment_dict = {
                    "随机裁剪+颜色抖动": transforms.Compose([
                        transforms.RandomResizedCrop(32),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                    ]),
                    "随机旋转+高斯模糊": transforms.Compose([
                        transforms.RandomRotation(15),
                        transforms.GaussianBlur(kernel_size=3)
                    ]),
                    "灰度+水平翻转": transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.RandomHorizontalFlip()
                    ])
                }
                
                model = SimCLREncoder()
                if (MODEL_PATH / "simclr_model.pth").exists():
                    model.load_state_dict(torch.load(MODEL_PATH / "simclr_model.pth", map_location=device))
                model.to(device)
                model.eval()
                
                all_combinations = []
                labels = []
                for i, aug_i in enumerate(aug_options):
                    for j, aug_j in enumerate(aug_options):
                        if i <= j:
                            aug_fn_i = augment_dict[aug_i]
                            aug_fn_j = augment_dict[aug_j]
                            with torch.no_grad():
                                zi = model(aug_fn_i(img_tensor))
                                zj = model(aug_fn_j(img_tensor))
                            sim = cosine_similarity(zi, zj)
                            all_combinations.append(sim)
                            labels.append(f"{aug_i}\nvs\n{aug_j}")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(labels, all_combinations)
                ax.set_ylabel('余弦相似度')
                ax.set_title('不同增强组合的相似度对比')
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                plt.close()
        
        with tab3:
            st.subheader("训练前后效果对比")
            st.markdown("""
            ### 📝 功能说明

            对比旋转预测模型的编码器在训练前（随机初始化）和训练后在CIFAR-10测试集上的t-SNE特征分布。颜色表示原始图片类别（飞机、汽车、鸟、猫等）。

            **预期效果**：
            - **训练前**：各类别的特征混杂在一起，没有明显的聚类，像一盘散沙
            - **训练后**：同类别的特征会聚集在一起，不同类别分开形成清晰的簇

            **为什么重要**：这证明了自监督学习确实让模型学到了有用的特征表示！
            """)
            
            if st.button("生成训练前后对比"):
                test_loader = DataLoader(test_dataset, batch_size=32)
                
                model_before = RotationCNN()
                model_before.to(device)
                model_before.eval()
                
                model_after = RotationCNN()
                if (MODEL_PATH / "rotation_model.pth").exists():
                    model_after.load_state_dict(torch.load(MODEL_PATH / "rotation_model.pth", map_location=device))
                model_after.to(device)
                model_after.eval()
                
                features_before = []
                features_after = []
                labels = []
                
                for images, lbls in test_loader:
                    images = images.to(device)
                    with torch.no_grad():
                        features_before.extend(model_before.get_features(images).detach().cpu().numpy())
                        features_after.extend(model_after.get_features(images).detach().cpu().numpy())
                    labels.extend(lbls.numpy())
                    if len(labels) >= 500:
                        break
                
                features_before = np.array(features_before)[:500]
                features_after = np.array(features_after)[:500]
                labels = np.array(labels)[:500]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                tsne = TSNE(n_components=2, random_state=42)
                feat_before_2d = tsne.fit_transform(features_before)
                scatter1 = ax1.scatter(feat_before_2d[:, 0], feat_before_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
                ax1.legend(handles=scatter1.legend_elements()[0], labels=class_names, title='类别')
                ax1.set_title('训练前（随机初始化）')
                
                feat_after_2d = tsne.fit_transform(features_after)
                scatter2 = ax2.scatter(feat_after_2d[:, 0], feat_after_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
                ax2.legend(handles=scatter2.legend_elements()[0], labels=class_names, title='类别')
                ax2.set_title('训练后')
                
                st.pyplot(fig)
                plt.close()

if __name__ == "__main__":
    main()
