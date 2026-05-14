import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import joblib
import matplotlib.font_manager as fm

_font_candidates = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
_available = [f.name for f in fm.fontManager.ttflist]
matplotlib.rcParams['font.sans-serif'] = [f for f in _font_candidates if f in _available] or ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import seaborn as sns
import pandas as pd
import time
from sklearn.datasets import load_digits, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
from skimage.transform import resize
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR.mkdir(exist_ok=True)

# 检查依赖
def check_dependencies():
    try:
        import torch
        import torchvision
        import sklearn
        import skimage
        import streamlit
        import matplotlib
        import seaborn
        import numpy
        import pandas
        from PIL import Image
        import scipy
        return True
    except ImportError as e:
        st.error(f"缺少依赖库: {e}")
        st.info("请运行: pip install torch torchvision scikit-learn scikit-image streamlit matplotlib seaborn pandas pillow scipy")
        return False

# 主应用
def main():
    st.set_page_config(
        page_title="计算机视觉教学演示",
        page_icon="👁️",
        layout="wide"
    )

    st.title("神经网络模型演示")
    st.sidebar.title("功能导航")

    # 功能选择
    feature = st.sidebar.selectbox(
        "选择功能",
        ["功能1: HOG + BoW + SVM 图像分类",
         "功能2: 反向传播 + Space Warping",
         "功能3: CNN 训练与测试",
         "功能4: 对比不同深度 ResNet"]
    )

    if feature == "功能1: HOG + BoW + SVM 图像分类":
        st.header("HOG + Bag of Words + SVM 图像分类")

        st.info("""
        **🎯 任务目标**：使用经典的计算机视觉算法组合（HOG特征提取 + 词袋模型 + SVM分类器）进行图像分类。

        **📊 主要控件说明**：
        - 「选择数据集」：可选择内置的手写数字数据集或上传自定义图像
        - 「训练模型」：训练SVM分类器，训练完成后自动保存模型
        - 「加载已训练模型」：直接加载之前保存的模型，无需重新训练

        **👀 结果展示**：
        - 训练后显示准确率、分类报告和混淆矩阵
        - 支持上传图片或随机选择图片进行预测
        - 显示真实标签与预测标签的对比
        """)

        # 数据集选择
        dataset_option = st.sidebar.radio("选择数据集", ["sklearn digits (8x8)", "上传自定义数据集"])

        # 全局变量
        if 'bow_model' not in st.session_state:
            st.session_state.bow_model = None
        if 'svm_model' not in st.session_state:
            st.session_state.svm_model = None
        if 'kmeans' not in st.session_state:
            st.session_state.kmeans = None
        if 'train_images' not in st.session_state:
            st.session_state.train_images = None
        if 'train_labels' not in st.session_state:
            st.session_state.train_labels = None

        # 提取HOG特征
        def extract_hog_features(images):
            features = []
            for img in images:
                if len(img.shape) == 3:
                    img = img.mean(axis=2)
                fd = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
                features.append(fd)
            return np.array(features)

        # 构建词袋模型
        def build_bow(features, n_clusters=50):
            # 随机采样特征
            sample_indices = np.random.choice(len(features), min(1000, len(features)), replace=False)
            sampled_features = features[sample_indices]

            # KMeans聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(sampled_features)
            return kmeans

        # 提取词袋直方图
        def extract_bow_features(features, kmeans):
            bow_features = []
            for feat in features:
                # 计算特征到每个聚类中心的距离
                distances = cdist([feat], kmeans.cluster_centers_)
                # 找到最近的聚类中心
                nearest = np.argmin(distances, axis=1)
                # 构建直方图
                hist, _ = np.histogram(nearest, bins=kmeans.n_clusters)
                # 归一化
                hist = hist / np.sum(hist)
                bow_features.append(hist)
            return np.array(bow_features)

        if dataset_option == "sklearn digits (8x8)":
            # 加载digits数据集
            digits = load_digits()
            images = digits.images
            labels = digits.target

            st.write(f"数据集大小: {len(images)} 张图像")
            st.write(f"图像尺寸: {images[0].shape}")

            # 显示样例图像
            st.subheader("样例图像")
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            for i, ax in enumerate(axes.flat):
                ax.imshow(images[i], cmap='gray')
                ax.set_title(f"标签: {labels[i]}")
                ax.axis('off')
            st.pyplot(fig)

            # 训练按钮
            if st.sidebar.button("训练模型"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("提取HOG特征...")
                hog_features = extract_hog_features(images)
                progress_bar.progress(25)

                status_text.text("构建词袋模型...")
                kmeans = build_bow(hog_features)
                st.session_state.kmeans = kmeans
                progress_bar.progress(50)

                status_text.text("提取词袋特征...")
                bow_features = extract_bow_features(hog_features, kmeans)
                progress_bar.progress(75)

                status_text.text("训练SVM模型...")
                # 划分训练测试集
                X_train, X_test, y_train, y_test = train_test_split(bow_features, labels, test_size=0.2, random_state=42)

                # 训练SVM
                svm = LinearSVC(random_state=42)
                svm.fit(X_train, y_train)
                st.session_state.svm_model = svm

                # 测试
                y_pred = svm.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                progress_bar.progress(100)
                status_text.text("训练完成！")

                # 显示结果
                st.subheader("训练结果")
                st.write(f"准确率: {accuracy:.4f}")

                st.subheader("分类报告")
                report = classification_report(y_test, y_pred)
                st.text(report)

                st.subheader("混淆矩阵")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
                st.pyplot(fig)

                # 保存训练数据
                st.session_state.train_images = images
                st.session_state.train_labels = labels

                # 保存模型
                joblib.dump(svm, MODEL_DIR / "svm_model.pkl")
                joblib.dump(kmeans, MODEL_DIR / "kmeans_model.pkl")
                st.success("模型已保存到 models/ 文件夹")

            # 加载模型按钮
            if st.sidebar.button("加载已训练模型"):
                svm_path = MODEL_DIR / "svm_model.pkl"
                kmeans_path = MODEL_DIR / "kmeans_model.pkl"

                if svm_path.exists() and kmeans_path.exists():
                    st.session_state.svm_model = joblib.load(svm_path)
                    st.session_state.kmeans = joblib.load(kmeans_path)
                    # 加载训练数据（用于随机预测）
                    digits = load_digits()
                    st.session_state.train_images = digits.images
                    st.session_state.train_labels = digits.target
                    st.success("模型加载成功！")
                else:
                    st.error("模型文件不存在，请先训练模型")

        else:  # 上传自定义数据集
            st.write("请上传两个类别的图像，每个类别至少20张")

            class1_files = st.file_uploader("上传类别1图像", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
            class2_files = st.file_uploader("上传类别2图像", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

            if len(class1_files) >= 20 and len(class2_files) >= 20:
                # 加载图像
                images = []
                labels = []

                for file in class1_files:
                    img = Image.open(file).convert('L')
                    img = resize(np.array(img), (32, 32))
                    images.append(img)
                    labels.append(0)

                for file in class2_files:
                    img = Image.open(file).convert('L')
                    img = resize(np.array(img), (32, 32))
                    images.append(img)
                    labels.append(1)

                images = np.array(images)
                labels = np.array(labels)

                st.write(f"数据集大小: {len(images)} 张图像")
                st.write(f"图像尺寸: {images[0].shape}")

                # 显示样例图像
                st.subheader("样例图像")
                fig, axes = plt.subplots(2, 5, figsize=(10, 4))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(images[i], cmap='gray')
                    ax.set_title(f"类别: {labels[i]}")
                    ax.axis('off')
                st.pyplot(fig)

                # 训练按钮
                if st.sidebar.button("训练模型"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("提取HOG特征...")
                    hog_features = extract_hog_features(images)
                    progress_bar.progress(25)

                    status_text.text("构建词袋模型...")
                    kmeans = build_bow(hog_features)
                    st.session_state.kmeans = kmeans
                    progress_bar.progress(50)

                    status_text.text("提取词袋特征...")
                    bow_features = extract_bow_features(hog_features, kmeans)
                    progress_bar.progress(75)

                    status_text.text("训练SVM模型...")
                    # 划分训练测试集
                    X_train, X_test, y_train, y_test = train_test_split(bow_features, labels, test_size=0.2, random_state=42)

                    # 训练SVM
                    svm = LinearSVC(random_state=42)
                    svm.fit(X_train, y_train)
                    st.session_state.svm_model = svm

                    # 测试
                    y_pred = svm.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    progress_bar.progress(100)
                    status_text.text("训练完成！")

                    # 显示结果
                    st.subheader("训练结果")
                    st.write(f"准确率: {accuracy:.4f}")

                    st.subheader("分类报告")
                    report = classification_report(y_test, y_pred)
                    st.text(report)

                    st.subheader("混淆矩阵")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('预测标签')
                    ax.set_ylabel('真实标签')
                    st.pyplot(fig)

                    # 保存训练数据
                    st.session_state.train_images = images
                    st.session_state.train_labels = labels
            else:
                st.warning("请上传至少20张图像到每个类别")

        # 单张图片预测
        st.subheader("单张图片预测")
        uploaded_file = st.file_uploader("上传图片进行预测", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None and st.session_state.svm_model is not None and st.session_state.kmeans is not None:
            # 加载并预处理图像
            img = Image.open(uploaded_file).convert('L')

            # 调整尺寸
            if st.session_state.train_images is not None:
                target_size = st.session_state.train_images[0].shape[:2]
                img = resize(np.array(img), target_size)
            else:
                img = resize(np.array(img), (8, 8))  # 默认digits尺寸

            # 显示图像
            st.image(img, caption="上传的图像", width=150)

            # 提取特征
            hog_feature = extract_hog_features([img])[0]
            bow_feature = extract_bow_features([hog_feature], st.session_state.kmeans)[0]

            # 预测
            prediction = st.session_state.svm_model.predict([bow_feature])[0]

            # 显示结果
            st.write(f"预测结果: {prediction}")
        elif uploaded_file is not None:
            st.warning("请先训练模型")

        # 随机图片预测
        st.subheader("随机图片预测")
        if st.session_state.svm_model is not None and st.session_state.kmeans is not None and st.session_state.train_images is not None:
            # 初始化随机索引
            if 'random_idx_1' not in st.session_state:
                st.session_state.random_idx_1 = 0

            if st.button("随机选择图片"):
                st.session_state.random_idx_1 = np.random.randint(0, len(st.session_state.train_images))

            # 获取随机图片和标签
            random_img = st.session_state.train_images[st.session_state.random_idx_1]
            true_label = st.session_state.train_labels[st.session_state.random_idx_1]

            # 显示图像（归一化到 [0, 1] 范围）
            img_display = random_img / random_img.max() if random_img.max() > 0 else random_img
            st.image(img_display, caption=f"随机图像 (索引: {st.session_state.random_idx_1})", width=150)

            # 提取特征并预测
            hog_feature = extract_hog_features([random_img])[0]
            bow_feature = extract_bow_features([hog_feature], st.session_state.kmeans)[0]
            prediction = st.session_state.svm_model.predict([bow_feature])[0]

            # 显示对比结果
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="真实标签", value=true_label)
            with col2:
                st.metric(label="预测标签", value=prediction)
            with col3:
                if prediction == true_label:
                    st.success("✅ 预测成功")
                else:
                    st.error("❌ 预测失败")
        else:
            st.warning("请先训练模型")
    elif feature == "功能2: 反向传播 + Space Warping":
        st.header("反向传播演示 + Space Warping 可视化")

        st.info("""
        **🎯 任务目标**：直观展示神经网络的核心概念——反向传播和特征空间变换。

        **📊 主要控件说明**：
        - 「选择模式」：三种演示模式供选择
          - 模式A：理解计算图和梯度传播的基本原理
          - 模式B：观察神经网络如何"扭曲"数据空间使非线性数据变得线性可分
          - 模式C：对比不同权重初始化方法对训练的影响

        **👀 结果展示**：
        - 模式A：显示计算图结构和各节点的梯度值
        - 模式B：实时展示训练过程中原始空间和隐藏层特征空间的变化
        - 模式C：对比不同初始化方法的损失曲线和最终准确率
        """)

        # 模式选择
        mode = st.sidebar.radio("选择模式", ["模式A: 简单计算图梯度演示", "模式B: 2D神经网络Space Warping", "模式C: 权重初始化对比"])

        if mode == "模式A: 简单计算图梯度演示":
            st.subheader("简单计算图梯度演示")
            st.write("计算图: f = (x + y) * z")

            # 用户输入
            x = st.slider("x", -10.0, 10.0, 1.0)
            y = st.slider("y", -10.0, 10.0, 2.0)
            z = st.slider("z", -10.0, 10.0, 3.0)

            # 前向传播
            if st.button("前向传播"):
                f = (x + y) * z
                st.write(f"f = (x + y) * z = ({x} + {y}) * {z} = {f}")

            # 反向传播
            if st.button("反向传播"):
                # 计算梯度
                df_dz = x + y
                df_dx = z
                df_dy = z

                st.write(f"df/dz = x + y = {df_dz}")
                st.write(f"df/dx = z = {df_dx}")
                st.write(f"df/dy = z = {df_dy}")

                # 绘制计算图
                fig, ax = plt.subplots(figsize=(10, 6))

                # 节点位置
                nodes = {
                    'x': (1, 2),
                    'y': (1, 1),
                    'z': (3, 1.5),
                    'add': (2, 1.5),
                    'mul': (3.5, 1.5),
                    'f': (4.5, 1.5)
                }

                # 绘制节点
                for node, pos in nodes.items():
                    ax.scatter(pos[0], pos[1], s=200, color='lightblue', edgecolor='black')
                    ax.text(pos[0], pos[1], node, ha='center', va='center', fontsize=12)

                # 绘制边
                edges = [
                    ('x', 'add'),
                    ('y', 'add'),
                    ('add', 'mul'),
                    ('z', 'mul'),
                    ('mul', 'f')
                ]

                for edge in edges:
                    start = nodes[edge[0]]
                    end = nodes[edge[1]]
                    ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)

                # 标注梯度
                ax.text(1, 2.2, f"df/dx = {df_dx}", ha='center', color='red')
                ax.text(1, 0.8, f"df/dy = {df_dy}", ha='center', color='red')
                ax.text(3, 1.7, f"df/dz = {df_dz}", ha='center', color='red')

                ax.set_xlim(0.5, 5)
                ax.set_ylim(0.5, 2.5)
                ax.axis('off')
                st.pyplot(fig)

        elif mode == "模式B: 2D神经网络Space Warping":
            st.subheader("2D神经网络Space Warping 可视化")
            st.write("演示带有ReLU激活的隐藏层如何将原始线性不可分的数据'扭曲'为线性可分")

            # 参数设置
            n_hidden = st.sidebar.slider("隐藏层神经元数", 2, 20, 5)
            learning_rate = st.sidebar.slider("学习率", 0.01, 1.0, 0.1)
            epochs = st.sidebar.slider("训练轮数", 50, 500, 150)

            # 生成数据集
            dataset = st.sidebar.radio("选择数据集", ["moons", "circles"])
            if dataset == "moons":
                X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
            else:
                X, y = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)

            # 神经网络类
            class SimpleNN(nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim):
                    super(SimpleNN, self).__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(hidden_dim, output_dim)

                def forward(self, x):
                    x = self.fc1(x)
                    self.hidden_output = self.relu(x)
                    x = self.fc2(x)
                    return x

                def get_hidden_output(self, x):
                    self.forward(x)
                    return self.hidden_output

            # 训练函数
            def train(model, X, y, epochs, learning_rate):
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)

                X_tensor = torch.FloatTensor(X)
                y_tensor = torch.LongTensor(y)

                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()

                    if (epoch + 1) % 10 == 0:
                        yield epoch + 1, model, loss.item()

            # 绘制决策边界
            def plot_decision_boundary(model, X, y, ax):
                h = 0.02
                x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
                y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

                Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
                Z = torch.argmax(Z, dim=1).detach().numpy()
                Z = Z.reshape(xx.shape)

                ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
                ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
                ax.set_title("原始空间决策边界")

            # 绘制隐藏层特征空间
            def plot_hidden_space(model, X, y, ax):
                hidden_output = model.get_hidden_output(torch.FloatTensor(X)).detach().numpy()

                if hidden_output.shape[1] > 2:
                    # 使用PCA降维
                    pca = PCA(n_components=2)
                    hidden_2d = pca.fit_transform(hidden_output)
                else:
                    hidden_2d = hidden_output

                ax.scatter(hidden_2d[:, 0], hidden_2d[:, 1], c=y, edgecolors='k', cmap='coolwarm')
                ax.set_title("隐藏层特征空间")

            # 开始训练按钮
            if st.sidebar.button("开始训练/重绘"):
                # 初始化模型
                model = SimpleNN(2, n_hidden, 2)

                # 显示训练过程
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 创建图表占位符
                chart_placeholder = st.empty()

                for epoch, model, loss in train(model, X, y, epochs, learning_rate):
                    progress = epoch / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"训练中... 第 {epoch}/{epochs} 轮, 损失: {loss:.4f}")

                    # 绘制图表
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    plot_decision_boundary(model, X, y, ax1)
                    plot_hidden_space(model, X, y, ax2)
                    plt.tight_layout()

                    # 更新图表
                    chart_placeholder.pyplot(fig)

                progress_bar.progress(1)
                status_text.text("训练完成！")

                # 保存模型
                st.session_state.nn_model = model

            # 显示初始数据
            st.subheader("初始数据")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
            ax.set_title("原始数据集")
            st.pyplot(fig)

            # 解释说明
            st.subheader("Space Warping 概念")
            st.write("1. 原始空间：数据是线性不可分的（如 moons 或 circles 数据集）")
            st.write("2. 隐藏层变换：通过 ReLU 激活函数，神经网络将数据映射到高维空间")
            st.write("3. 新特征空间：在这个空间中，数据变得线性可分")
            st.write("4. 输出层：在新空间中使用线性分类器（如 Softmax）进行分类")

        elif mode == "模式C: 权重初始化对比":
            st.subheader("权重初始化对比")
            st.write("比较不同权重初始化方法对神经网络训练的影响")

            # 按钮
            if st.button("运行初始化对比实验"):
                # 生成数据集
                X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # 超参数
                epochs = 150
                lr = 0.1
                input_dim = 2
                hidden_dim = 10
                output_dim = 2

                # 权重初始化方法
                initializations = {
                    "太小": lambda fan_in, fan_out: 0.01 * np.random.randn(fan_in, fan_out),
                    "太大": lambda fan_in, fan_out: 0.05 * np.random.randn(fan_in, fan_out),
                    "Xavier": lambda fan_in, fan_out: np.random.uniform(-np.sqrt(6 / (fan_in + fan_out)), np.sqrt(6 / (fan_in + fan_out)), (fan_in, fan_out)),
                    "MSRA (He)": lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)
                }

                # 存储结果
                all_losses = {}
                all_accuracies = {}

                # 总进度条
                total_progress = st.progress(0)
                status_text = st.empty()

                # 训练函数
                def train_with_initialization(X_train, y_train, X_test, y_test, weight_init, input_dim, hidden_dim, output_dim, epochs, lr):
                    # 初始化权重
                    W1 = weight_init(input_dim, hidden_dim)
                    b1 = np.zeros((1, hidden_dim))
                    W2 = weight_init(hidden_dim, output_dim)
                    b2 = np.zeros((1, output_dim))

                    losses = []

                    for epoch in range(epochs):
                        # 前向传播
                        z1 = np.dot(X_train, W1) + b1
                        a1 = np.maximum(0, z1)  # ReLU
                        z2 = np.dot(a1, W2) + b2

                        #  softmax
                        exp_scores = np.exp(z2)
                        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                        # 计算损失
                        correct_logprobs = -np.log(probs[range(len(y_train)), y_train])
                        loss = np.mean(correct_logprobs)
                        losses.append(loss)

                        # 反向传播
                        delta3 = probs
                        delta3[range(len(y_train)), y_train] -= 1
                        delta2 = np.dot(delta3, W2.T) * (a1 > 0)

                        # 梯度
                        dW2 = np.dot(a1.T, delta3)
                        db2 = np.sum(delta3, axis=0, keepdims=True)
                        dW1 = np.dot(X_train.T, delta2)
                        db1 = np.sum(delta2, axis=0, keepdims=True)

                        # 更新权重
                        W1 -= lr * dW1
                        b1 -= lr * db1
                        W2 -= lr * dW2
                        b2 -= lr * db2

                    # 测试准确率
                    z1 = np.dot(X_test, W1) + b1
                    a1 = np.maximum(0, z1)
                    z2 = np.dot(a1, W2) + b2
                    exp_scores = np.exp(z2)
                    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                    y_pred = np.argmax(probs, axis=1)
                    accuracy = np.mean(y_pred == y_test)

                    return losses, accuracy

                # 对每种初始化方法进行训练
                for i, (name, weight_init) in enumerate(initializations.items()):
                    status_text.text(f"训练 {name} 初始化...")

                    # 训练
                    losses, accuracy = train_with_initialization(
                        X_train, y_train, X_test, y_test,
                        weight_init, input_dim, hidden_dim, output_dim,
                        epochs, lr
                    )

                    all_losses[name] = losses
                    all_accuracies[name] = accuracy

                    # 更新进度
                    total_progress.progress((i + 1) / len(initializations))

                status_text.text("训练完成！")

                # 绘制对比图表 - 一行两张
                st.subheader("权重初始化对比结果")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # 损失曲线对比
                for name, losses in all_losses.items():
                    ax1.plot(range(epochs), losses, label=name)
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.set_title("不同初始化方法的损失曲线")
                ax1.legend()

                # 准确率柱状图
                names = list(all_accuracies.keys())
                accuracies = list(all_accuracies.values())
                ax2.bar(names, accuracies)
                ax2.set_xlabel("初始化方法")
                ax2.set_ylabel("测试准确率")
                ax2.set_title("不同初始化方法的测试准确率")
                ax2.set_ylim(0, 1)
                for i, acc in enumerate(accuracies):
                    ax2.text(i, acc + 0.02, f"{acc:.4f}", ha='center')

                plt.tight_layout()
                st.pyplot(fig)
    elif feature == "功能3: CNN 训练与测试":
        st.header("CNN 训练与测试（LeNet-5 风格）")

        st.info("""
        **🎯 任务目标**：训练经典的LeNet-5卷积神经网络进行图像分类，并探索数据增强对模型性能的影响。

        **📊 主要控件说明**：
        - 「训练轮数」：设置模型训练的迭代次数
        - 「批量大小」：每次训练时输入模型的图像数量
        - 「学习率」：控制模型参数更新的步长
        - 「启用数据增强对比模式」：同时训练两个模型，对比有无数据增强的效果
        - 「加载已训练模型」：直接加载之前保存的模型
        - 「开始训练」：开始训练模型

        **👀 结果展示**：
        - 训练过程中实时显示损失曲线和测试准确率
        - 数据增强对比模式下展示两个模型的性能差异
        - 支持上传图片或随机选择图片进行预测
        - 显示各类别的预测概率
        """)

        # 数据集选择
        dataset = st.sidebar.selectbox("选择数据集", ["CIFAR-10"])

        # 训练参数
        epochs = st.sidebar.slider("训练轮数", 1, 10, 2)
        batch_size = st.sidebar.slider("批量大小", 32, 128, 64)
        learning_rate = st.sidebar.slider("学习率", 0.001, 0.1, 0.01)
        use_aug_compare = st.sidebar.checkbox("启用数据增强对比模式")

        # LeNet-5 模型定义（适配CIFAR-10）
        class LeNet5(nn.Module):
            def __init__(self, in_channels=1, num_classes=10):
                super(LeNet5, self).__init__()
                self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
                # 为了兼容两种输入尺寸，使用自适应池化
                self.avg_pool = nn.AdaptiveAvgPool2d((5, 5))
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, num_classes)

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.relu(self.conv2(x))
                x = self.avg_pool(x)
                x = x.view(-1, 16 * 5 * 5)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        # 数据加载
        def load_data(dataset_name, batch_size, use_aug_compare=False):
            try:
                if dataset_name == "MNIST":
                    # 基础变换
                    base_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
                    # 数据增强变换
                    aug_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])

                    testset = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=base_transform)
                    in_channels = 1
                else:  # CIFAR-10
                    # 基础变换
                    base_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
                    # 数据增强变换
                    aug_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

                    # 使用本地CIFAR-10数据集，设置download=True以便自动下载
                    testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=base_transform)
                    in_channels = 3

                # 测试加载器
                testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

                if use_aug_compare:
                    # 无数据增强的训练集
                    if dataset_name == "MNIST":
                        trainset_base = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=base_transform)
                    else:
                        trainset_base = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=base_transform)

                    # 有数据增强的训练集
                    if dataset_name == "MNIST":
                        trainset_aug = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=aug_transform)
                    else:
                        trainset_aug = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=aug_transform)

                    # 创建两个训练加载器
                    trainloader_base = torch.utils.data.DataLoader(trainset_base, batch_size=batch_size, shuffle=True)
                    trainloader_aug = torch.utils.data.DataLoader(trainset_aug, batch_size=batch_size, shuffle=True)

                    return trainloader_base, trainloader_aug, testloader, in_channels
                else:
                    # 只返回一个训练加载器（无数据增强）
                    if dataset_name == "MNIST":
                        trainset = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=base_transform)
                    else:
                        trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=base_transform)

                    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
                    return trainloader, testloader, in_channels
            except Exception as e:
                st.error(f"数据集加载失败: {e}")
                st.info("请尝试以下解决方案:")
                st.info("1. 确保CIFAR-10数据集已解压到 data 目录")
                st.info("2. 检查数据集文件是否完整")
                if use_aug_compare:
                    return None, None, None, None
                else:
                    return None, None, None

        # 训练函数
        def train_cnn(model, trainloader, testloader, epochs, learning_rate):
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_losses = []
            test_accuracies = []

            for epoch in range(epochs):
                running_loss = 0.0
                model.train()

                for i, (inputs, labels) in enumerate(trainloader):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    # 每100个批次更新一次
                    if i % 100 == 99:
                        avg_loss = running_loss / 100
                        train_losses.append(avg_loss)
                        running_loss = 0.0

                        # 测试准确率
                        correct = 0
                        total = 0
                        model.eval()
                        with torch.no_grad():
                            for inputs, labels in testloader:
                                outputs = model(inputs)
                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()

                        accuracy = 100 * correct / total
                        test_accuracies.append(accuracy)
                        model.train()

                        yield epoch + 1, i + 1, avg_loss, accuracy, train_losses, test_accuracies

        # 加载模型按钮
        if st.sidebar.button("加载已训练模型"):
            model_path = MODEL_DIR / "lenet5_model.pth"

            if model_path.exists():
                # 初始化模型
                in_channels = 3 if dataset == "CIFAR-10" else 1
                model = LeNet5(in_channels=in_channels)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                st.session_state.cnn_model = model
                st.session_state.cnn_dataset = dataset
                st.success("模型加载成功！")
            else:
                st.error("模型文件不存在，请先训练模型")

        # 开始训练按钮
        if st.sidebar.button("开始训练"):
            if use_aug_compare:
                # 加载数据（带数据增强对比）
                trainloader_base, trainloader_aug, testloader, in_channels = load_data(dataset, batch_size, use_aug_compare=True)

                # 检查数据加载是否成功
                if trainloader_base is None or trainloader_aug is None or testloader is None:
                    st.error("数据加载失败，无法开始训练")
                else:
                    # 初始化两个模型
                    model_base = LeNet5(in_channels=in_channels)
                    model_aug = LeNet5(in_channels=in_channels)

                    # 显示训练过程
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # 创建图表占位符（无数据增强和有数据增强各一个）
                    base_chart_placeholder = st.empty()
                    aug_chart_placeholder = st.empty()

                    # 训练无数据增强的模型
                    status_text.text("训练无数据增强的模型...")
                    total_steps = epochs * len(trainloader_base)
                    step = 0
                    train_losses_base = []
                    test_accuracies_base = []

                    for epoch, batch, loss, accuracy, losses, accuracies in train_cnn(model_base, trainloader_base, testloader, epochs, learning_rate):
                        step += 100
                        progress = min(step / (total_steps * 2), 0.5)
                        progress_bar.progress(progress)
                        status_text.text(f"训练无数据增强的模型... 第 {epoch}/{epochs} 轮, 批次: {batch}/{len(trainloader_base)}")

                        train_losses_base = losses
                        test_accuracies_base = accuracies

                        # 更新无数据增强的图表 - 一行两张
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
                        # 损失曲线
                        ax1.plot(range(len(train_losses_base)), train_losses_base)
                        ax1.set_xlabel("批次")
                        ax1.set_ylabel("损失")
                        ax1.set_title("无数据增强 - 损失曲线")
                        # 准确率曲线
                        ax2.plot(range(len(test_accuracies_base)), test_accuracies_base)
                        ax2.set_xlabel("批次")
                        ax2.set_ylabel("准确率")
                        ax2.set_title("无数据增强 - 准确率曲线")
                        plt.tight_layout()
                        base_chart_placeholder.pyplot(fig)
                        plt.close()

                    # 训练有数据增强的模型
                    status_text.text("训练有数据增强的模型...")
                    total_steps = epochs * len(trainloader_aug)
                    step = 0
                    train_losses_aug = []
                    test_accuracies_aug = []

                    for epoch, batch, loss, accuracy, losses, accuracies in train_cnn(model_aug, trainloader_aug, testloader, epochs, learning_rate):
                        step += 100
                        progress = 0.5 + min(step / (total_steps * 2), 0.5)
                        progress_bar.progress(progress)
                        status_text.text(f"训练有数据增强的模型... 第 {epoch}/{epochs} 轮, 批次: {batch}/{len(trainloader_aug)}")

                        train_losses_aug = losses
                        test_accuracies_aug = accuracies

                        # 更新有数据增强的图表 - 一行两张
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
                        # 损失曲线
                        ax1.plot(range(len(train_losses_aug)), train_losses_aug)
                        ax1.set_xlabel("批次")
                        ax1.set_ylabel("损失")
                        ax1.set_title("有数据增强 - 损失曲线")
                        # 准确率曲线
                        ax2.plot(range(len(test_accuracies_aug)), test_accuracies_aug)
                        ax2.set_xlabel("批次")
                        ax2.set_ylabel("准确率")
                        ax2.set_title("有数据增强 - 准确率曲线")
                        plt.tight_layout()
                        aug_chart_placeholder.pyplot(fig)
                        plt.close()

                    progress_bar.progress(1)
                    status_text.text("训练完成！")

                    # 比较结果
                    st.subheader("数据增强对比结果")

                    # 绘制对比图表 - 一行两张
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                    # 损失曲线对比
                    ax1.plot(range(len(train_losses_base)), train_losses_base, label="无数据增强")
                    ax1.plot(range(len(train_losses_aug)), train_losses_aug, label="有数据增强")
                    ax1.set_xlabel("批次")
                    ax1.set_ylabel("损失")
                    ax1.set_title("损失曲线对比")
                    ax1.legend()

                    # 准确率曲线对比
                    ax2.plot(range(len(test_accuracies_base)), test_accuracies_base, label="无数据增强")
                    ax2.plot(range(len(test_accuracies_aug)), test_accuracies_aug, label="有数据增强")
                    ax2.set_xlabel("批次")
                    ax2.set_ylabel("准确率")
                    ax2.set_title("准确率曲线对比")
                    ax2.legend()

                    plt.tight_layout()
                    st.pyplot(fig)

                    # 显示最终准确率
                    final_acc_base = test_accuracies_base[-1] if test_accuracies_base else 0
                    final_acc_aug = test_accuracies_aug[-1] if test_accuracies_aug else 0

                    st.write(f"无数据增强的最终测试准确率: {final_acc_base:.2f}%")
                    st.write(f"有数据增强的最终测试准确率: {final_acc_aug:.2f}%")

                    # 保存模型
                    torch.save(model_base.state_dict(), MODEL_DIR / "lenet5_model_base.pth")
                    torch.save(model_aug.state_dict(), MODEL_DIR / "lenet5_model_aug.pth")
                    st.session_state.cnn_model = model_aug  # 默认保存有数据增强的模型
                    st.session_state.cnn_dataset = dataset
                    st.success("模型已保存到 models/ 文件夹")
            else:
                # 加载数据（无数据增强对比）
                trainloader, testloader, in_channels = load_data(dataset, batch_size)

                # 检查数据加载是否成功
                if trainloader is None or testloader is None:
                    st.error("数据加载失败，无法开始训练")
                else:
                    # 初始化模型
                    model = LeNet5(in_channels=in_channels)

                    # 显示训练过程
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # 创建图表占位符（一行两张）
                    chart_placeholder = st.empty()

                    total_steps = epochs * len(trainloader)
                    step = 0

                    for epoch, batch, loss, accuracy, train_losses, test_accuracies in train_cnn(model, trainloader, testloader, epochs, learning_rate):
                        step += 100
                        progress = min(step / total_steps, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"训练中... 第 {epoch}/{epochs} 轮, 批次: {batch}/{len(trainloader)}, 损失: {loss:.4f}, 测试准确率: {accuracy:.2f}%")

                        # 更新图表 - 一行两张
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
                        # 损失曲线
                        ax1.plot(range(len(train_losses)), train_losses)
                        ax1.set_xlabel("批次")
                        ax1.set_ylabel("损失")
                        ax1.set_title("损失曲线")
                        # 准确率曲线
                        ax2.plot(range(len(test_accuracies)), test_accuracies)
                        ax2.set_xlabel("批次")
                        ax2.set_ylabel("准确率")
                        ax2.set_title("准确率曲线")
                        plt.tight_layout()
                        chart_placeholder.pyplot(fig)
                        plt.close()

                    progress_bar.progress(1)
                    status_text.text("训练完成！")

                    # 保存模型
                    torch.save(model.state_dict(), MODEL_DIR / "lenet5_model.pth")
                    st.session_state.cnn_model = model
                    st.session_state.cnn_dataset = dataset
                    st.success("模型已保存到 models/ 文件夹")

        # 单张图片预测
        st.subheader("单张图片预测")
        uploaded_file = st.file_uploader("上传图片进行预测", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None and 'cnn_model' in st.session_state:
            # 加载并预处理图像
            img = Image.open(uploaded_file)

            # 预处理
            if st.session_state.cnn_dataset == "MNIST":
                transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                classes = [str(i) for i in range(10)]
            else:  # CIFAR-10
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

            img_tensor = transform(img).unsqueeze(0)

            # 显示图像
            st.image(img, caption="上传的图像", width=150)

            # 预测
            model = st.session_state.cnn_model
            model.eval()
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().numpy()
                predicted = torch.argmax(outputs, dim=1).item()

            # 显示结果
            st.write(f"预测结果: {classes[predicted]}")

            # 显示概率条形图
            st.subheader("类别概率")
            prob_df = pd.DataFrame({'类别': classes, '概率': probabilities})
            prob_df = prob_df.sort_values('概率', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='概率', y='类别', data=prob_df, ax=ax)
            ax.set_xlim(0, 1)
            st.pyplot(fig)
        elif uploaded_file is not None:
            st.warning("请先训练模型")

        # 随机图片预测
        st.subheader("随机图片预测")
        if 'cnn_model' in st.session_state and 'cnn_dataset' in st.session_state:
            # 加载测试数据集用于随机选择
            if st.session_state.cnn_dataset == "MNIST":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                testset = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)
                classes = [str(i) for i in range(10)]
            else:  # CIFAR-10
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform)
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

            # 初始化随机索引
            if 'random_idx_3' not in st.session_state:
                st.session_state.random_idx_3 = 0

            if st.button("随机选择图片"):
                st.session_state.random_idx_3 = np.random.randint(0, len(testset))

            # 获取随机图片和标签
            random_img, true_label = testset[st.session_state.random_idx_3]

            # 显示图像（裁剪到 [0, 1] 范围）
            img_np = random_img.permute(1, 2, 0).numpy()
            if st.session_state.cnn_dataset == "MNIST":
                img_np = img_np.squeeze()
            # 裁剪到 [0, 1] 范围
            img_np = np.clip(img_np, 0, 1)
            st.image(img_np, caption=f"随机图像 (索引: {st.session_state.random_idx_3})", width=150)

            # 预测
            model = st.session_state.cnn_model
            model.eval()
            with torch.no_grad():
                outputs = model(random_img.unsqueeze(0))
                probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().numpy()
                predicted = torch.argmax(outputs, dim=1).item()

            # 显示对比结果
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="真实标签", value=classes[true_label])
            with col2:
                st.metric(label="预测标签", value=classes[predicted])
            with col3:
                if predicted == true_label:
                    st.success("✅ 预测成功")
                else:
                    st.error("❌ 预测失败")

            # 显示概率条形图
            st.subheader("类别概率")
            prob_df = pd.DataFrame({'类别': classes, '概率': probabilities})
            prob_df = prob_df.sort_values('概率', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='概率', y='类别', data=prob_df, ax=ax)
            ax.set_xlim(0, 1)
            st.pyplot(fig)
        else:
            st.warning("请先训练模型")
    elif feature == "功能4: 对比不同深度 ResNet":
        st.header("对比不同深度预训练 ResNet 的性能")

        st.info("""
        **🎯 任务目标**：对比不同深度的预训练ResNet模型（18层、34层、50层）在图像分类任务上的性能差异。

        **📊 主要控件说明**：
        - 「开始评估」：对ResNet-18、ResNet-34、ResNet-50三个模型进行性能评估
        - 「选择模型」（预测时）：选择要使用的ResNet模型

        **👀 结果展示**：
        - 评估完成后显示各模型的准确率、推理时间和参数量
        - 以图表形式对比不同模型的性能差异
        - 支持上传图片或随机选择图片进行单张预测
        - 显示Top-5预测结果和对应的概率
        """)

        # 数据集选择
        dataset = st.sidebar.selectbox("选择数据集", ["CIFAR-10"])

        # 模型列表
        models_to_evaluate = ["ResNet-18", "ResNet-34", "ResNet-50"]

        # 加载数据
        def load_cifar10(batch_size=32):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # ResNet输入尺寸
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
            return testloader

        # 加载模型
        def load_model(model_name):
            if model_name == "ResNet-18":
                model = resnet18(pretrained=True)
            elif model_name == "ResNet-34":
                model = resnet34(pretrained=True)
            elif model_name == "ResNet-50":
                model = resnet50(pretrained=True)

            # 冻结卷积层
            for param in model.parameters():
                param.requires_grad = False

            # 替换全连接层
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10有10个类别

            return model

        # 计算参数量
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        # 评估模型
        def evaluate_model(model, testloader):
            model.eval()
            correct = 0
            total = 0
            inference_times = []

            with torch.no_grad():
                for i, (inputs, labels) in enumerate(testloader):
                    # 计算推理时间
                    start_time = time.time()
                    outputs = model(inputs)
                    end_time = time.time()
                    inference_times.append(end_time - start_time)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # 只评估100张图片的推理时间
                    if i * testloader.batch_size >= 100:
                        break

            accuracy = 100 * correct / total
            avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒

            return accuracy, avg_inference_time

        # 开始评估按钮
        if st.sidebar.button("开始评估"):
            # 加载测试数据
            testloader = load_cifar10(batch_size=32)

            # 评估结果
            results = []

            # 显示进度
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, model_name in enumerate(models_to_evaluate):
                status_text.text(f"评估 {model_name}...")

                # 加载模型
                model = load_model(model_name)

                # 计算参数量
                params = count_parameters(model)

                # 评估模型
                accuracy, avg_time = evaluate_model(model, testloader)

                # 保存结果
                results.append({
                    "模型": model_name,
                    "准确率": accuracy,
                    "平均推理时间 (ms)": avg_time,
                    "参数量 (M)": params / 1e6
                })

                # 更新进度
                progress_bar.progress((i + 1) / len(models_to_evaluate))

            status_text.text("评估完成！")

            # 保存结果到session_state
            st.session_state.resnet_results = results

            # 显示结果
            st.subheader("评估结果")
            df = pd.DataFrame(results)
            st.dataframe(df)

            # 绘制对比图表 - 一行两张
            st.subheader("模型性能对比")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # 准确率对比
            sns.barplot(x="模型", y="准确率", data=df, ax=ax1)
            ax1.set_ylim(0, 100)
            ax1.set_title("准确率对比")

            # 推理时间对比
            sns.barplot(x="模型", y="平均推理时间 (ms)", data=df, ax=ax2)
            ax2.set_title("推理时间对比")

            plt.tight_layout()
            st.pyplot(fig)

        # 单张图片预测
        st.subheader("单张图片预测")

        # 选择模型
        selected_model = st.selectbox("选择模型", models_to_evaluate)

        # 上传图片
        uploaded_file = st.file_uploader("上传图片进行预测", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # 加载模型
            model = load_model(selected_model)
            model.eval()

            # 预处理图像
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            img = Image.open(uploaded_file)
            img_tensor = transform(img).unsqueeze(0)

            # 显示图像
            st.image(img, caption="上传的图像", width=200)

            # 预测
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().numpy()
                top5_indices = np.argsort(probabilities)[::-1][:5]
                top5_probabilities = probabilities[top5_indices]

            # CIFAR-10类别
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            top5_classes = [classes[i] for i in top5_indices]

            # 显示Top-5结果
            st.subheader("Top-5 预测结果")
            for i, (cls, prob) in enumerate(zip(top5_classes, top5_probabilities)):
                st.write(f"{i+1}. {cls}: {prob:.4f}")

            # 绘制Top-5概率条形图
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top5_probabilities, y=top5_classes, ax=ax)
            ax.set_xlim(0, 1)
            st.pyplot(fig)

        # 随机图片预测
        st.subheader("随机图片预测")

        # 选择模型
        selected_model_pred = st.selectbox("选择模型", models_to_evaluate, key="resnet_select_pred")

        # CIFAR-10类别
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # 加载测试数据集用于随机选择
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform)

        # 初始化随机索引
        if 'random_idx_4' not in st.session_state:
            st.session_state.random_idx_4 = 0

        if st.button("随机选择图片"):
            st.session_state.random_idx_4 = np.random.randint(0, len(testset))

        # 获取随机图片和标签
        random_img, true_label = testset[st.session_state.random_idx_4]

        # 显示图像（裁剪到 [0, 1] 范围）
        img_np = random_img.permute(1, 2, 0).numpy()
        # 反归一化以便显示
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        st.image(img_np, caption=f"随机图像 (索引: {st.session_state.random_idx_4})", width=200)

        # 加载模型并预测
        model = load_model(selected_model_pred)
        model.eval()

        with torch.no_grad():
            outputs = model(random_img.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().numpy()
            predicted = torch.argmax(outputs, dim=1).item()

        # 显示对比结果
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="真实标签", value=classes[true_label])
        with col2:
            st.metric(label="预测标签", value=classes[predicted])
        with col3:
            if predicted == true_label:
                st.success("✅ 预测成功")
            else:
                st.error("❌ 预测失败")

        # 显示Top-5结果
        st.subheader("Top-5 预测结果")
        top5_indices = np.argsort(probabilities)[::-1][:5]
        top5_probabilities = probabilities[top5_indices]
        top5_classes = [classes[i] for i in top5_indices]

        for i, (cls, prob) in enumerate(zip(top5_classes, top5_probabilities)):
            st.write(f"{i+1}. {cls}: {prob:.4f}")

        # 绘制Top-5概率条形图
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top5_probabilities, y=top5_classes, ax=ax)
        ax.set_xlim(0, 1)
        st.pyplot(fig)

if __name__ == "__main__":
    if check_dependencies():
        main()
