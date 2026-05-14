import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import time
import os

_font_candidates = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
_available = [f.name for f in fm.fontManager.ttflist]
matplotlib.rcParams['font.sans-serif'] = [f for f in _font_candidates if f in _available] or ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

DATA_DIR = Path(__file__).parent / "data"

def _load_cifar10_numpy(n_train=None, n_test=None):
    train_set = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True)

    x_train = np.array(train_set.data, dtype=np.float32) / 255.0
    y_train = np.array(train_set.targets, dtype=np.int64)
    x_test = np.array(test_set.data, dtype=np.float32) / 255.0
    y_test = np.array(test_set.targets, dtype=np.int64)

    if n_train:
        x_train, y_train = x_train[:n_train], y_train[:n_train]
    if n_test:
        x_test, y_test = x_test[:n_test], y_test[:n_test]

    return x_train, y_train.reshape(-1, 1), x_test, y_test.reshape(-1, 1)

def load_cifar10():
    return _load_cifar10_numpy(n_train=1000, n_test=200)

def load_cifar10_full():
    return _load_cifar10_numpy(n_train=5000)

# 模块一：最小二乘线性回归演示
def linear_regression_demo():
    st.header("最小二乘线性回归演示")
    
    # 初始化数据
    if 'data_points' not in st.session_state:
        # 生成线性+噪声数据
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + np.random.normal(0, 2, 100)
        st.session_state.data_points = np.column_stack((x, y))
    
    # 侧边栏参数设置
    st.sidebar.subheader("参数设置")
    train_ratio = st.sidebar.slider("训练集比例", 0.5, 0.9, 0.7, 0.05)
    polynomial_order = st.sidebar.slider("多项式阶数", 1, 10, 1, 1)
    lambda_reg = st.sidebar.slider("L2 正则化强度 λ", 0.0, 5.0, 0.0, 0.1)
    
    # 数据点管理
    col1, col2 = st.columns(2)
    with col1:
        if st.button("重置数据"):
            np.random.seed(42)
            x = np.linspace(0, 10, 100)
            y = 2 * x + 1 + np.random.normal(0, 2, 100)
            st.session_state.data_points = np.column_stack((x, y))
    
    with col2:
        if st.button("添加随机点"):
            new_x = np.random.uniform(0, 10, 10)
            new_y = 2 * new_x + 1 + np.random.normal(0, 2, 10)
            new_points = np.column_stack((new_x, new_y))
            st.session_state.data_points = np.vstack((st.session_state.data_points, new_points))
    
    # 分割训练集和测试集
    data = st.session_state.data_points
    n = len(data)
    n_train = int(n * train_ratio)
    indices = np.random.permutation(n)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    x_train = data[train_indices, 0]
    y_train = data[train_indices, 1]
    x_test = data[test_indices, 0]
    y_test = data[test_indices, 1]
    
    # 多项式特征转换
    def polynomial_features(x, order):
        features = np.ones((len(x), 1))
        for i in range(1, order + 1):
            features = np.hstack((features, x[:, np.newaxis] ** i))
        return features
    
    X_train = polynomial_features(x_train, polynomial_order)
    X_test = polynomial_features(x_test, polynomial_order)
    
    # 最小二乘法闭式解（带正则化）
    I = np.eye(X_train.shape[1])
    I[0, 0] = 0  #  bias 项不正则化
    theta = np.linalg.inv(X_train.T @ X_train + lambda_reg * I) @ X_train.T @ y_train
    
    # 预测
    y_train_pred = X_train @ theta
    y_test_pred = X_test @ theta
    
    # 计算 MSE
    mse_train = np.mean((y_train - y_train_pred) ** 2)
    mse_test = np.mean((y_test - y_test_pred) ** 2)
    
    # 绘制图像
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制训练集和测试集
    ax.scatter(x_train, y_train, label='训练集', alpha=0.6)
    ax.scatter(x_test, y_test, label='测试集', alpha=0.6, marker='x')
    
    # 绘制拟合曲线
    x_plot = np.linspace(0, 10, 100)
    X_plot = polynomial_features(x_plot, polynomial_order)
    y_plot = X_plot @ theta
    ax.plot(x_plot, y_plot, 'r-', label='拟合曲线', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'多项式回归 (阶数={polynomial_order}, λ={lambda_reg:.2f})')
    ax.legend()
    
    st.pyplot(fig)
    
    # 显示 MSE
    st.write(f"训练集 MSE: {mse_train:.4f}")
    st.write(f"测试集 MSE: {mse_test:.4f}")

# 自定义 KNN 分类器
class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = []
        for x in X:
            # 计算欧氏距离
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # 排序并取前 k 个
            k_indices = np.argsort(distances)[:self.k]
            # 多数投票
            k_nearest_labels = self.y_train[k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            y_pred.append(most_common)
        return np.array(y_pred)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def kneighbors(self, X, return_distance=True):
        distances = []
        indices = []
        for x in X:
            # 计算欧氏距离
            dist = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # 排序并取前 k 个
            k_indices = np.argsort(dist)[:self.k]
            k_distances = dist[k_indices]
            distances.append(k_distances)
            indices.append(k_indices)
        if return_distance:
            return np.array(distances), np.array(indices)
        else:
            return np.array(indices)

# 模块二：KNN 分类器
def knn_classifier():
    st.header("KNN 分类器 (CIFAR-10)")
    
    # 加载 CIFAR-10 数据
    with st.spinner("加载 CIFAR-10 数据..."):
        x_train, y_train, x_test, y_test = load_cifar10()
    
    st.info("使用 CIFAR-10 数据集：1000 张训练图片，200 张测试图片")
    
    # 先显示一些样本图片看看数据是否正常
    st.subheader("样本图片预览")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(x_train[i], width=80, clamp=True)
            st.write(f"类别: {class_names[y_train[i][0]]}")
    
    # 数据预处理
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()
    
    # 侧边栏参数设置
    st.sidebar.subheader("KNN 参数")
    k_value = st.sidebar.slider("K 值", 1, 15, 5, 1)
    
    # KNN 训练按钮
    if st.button("训练 KNN 分类器"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 训练 KNN 分类器
        status_text.text("正在训练 KNN 分类器...")
        knn = KNN(k=k_value)
        knn.fit(x_train_flat, y_train_flat)
        progress_bar.progress(50)
        
        # 计算准确率
        status_text.text("正在计算准确率...")
        accuracy = knn.score(x_test_flat, y_test_flat)
        progress_bar.progress(100)
        
        st.success(f"训练完成！K 值 ({k_value}) 的测试集准确率: {accuracy:.4f}")
        st.session_state.knn_model = knn
        st.session_state.knn_accuracy = accuracy
    else:
        # 检查是否已训练
        if 'knn_model' in st.session_state:
            knn = st.session_state.knn_model
            accuracy = st.session_state.knn_accuracy
            st.write(f"当前 K 值 ({k_value}) 的测试集准确率: {accuracy:.4f}")
        else:
            st.write("点击'训练 KNN 分类器'按钮开始训练")
    
    # 不同 K 值的准确率对比
    st.subheader("不同 K 值的准确率对比")
    if st.button("评估不同 K 值"):
        k_values = [1, 3, 5, 7, 9, 11, 13, 15]
        accuracies = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, k in enumerate(k_values):
            status_text.text(f"正在评估 K={k}...")
            knn_temp = KNN(k=k)
            knn_temp.fit(x_train_flat, y_train_flat)
            acc = knn_temp.score(x_test_flat, y_test_flat)
            accuracies.append(acc)
            progress_bar.progress((i + 1) / len(k_values))
        
        status_text.text("评估完成！")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(k_values, accuracies)
        ax.set_xlabel('K 值')
        ax.set_ylabel('准确率')
        ax.set_title('不同 K 值的准确率对比')
        ax.set_ylim(0, 1)
        
        for i, acc in enumerate(accuracies):
            ax.text(k_values[i], acc + 0.01, f'{acc:.4f}', ha='center')
        
        st.pyplot(fig)
    
    # 随机选择测试图片并显示最近邻
    st.subheader("最近邻可视化")
    if st.button("随机选择测试图片"):
        # 随机选择一张测试图片
        idx = np.random.randint(0, len(x_test))
        test_img = x_test[idx]
        test_img_flat = x_test_flat[idx].reshape(1, -1)
        
        # 显示测试图片
        st.write(f"测试图片 (真实类别: {class_names[y_test_flat[idx]]})")
        st.image(test_img, width=150)
        
        # 为不同 K 值显示最近邻
        k_list = [1, 3, 5, 10]
        for k in k_list:
            knn_temp = KNN(k=k)
            knn_temp.fit(x_train_flat, y_train_flat)
            distances, indices = knn_temp.kneighbors(test_img_flat)
            
            st.write(f"\nK={k} 的最近邻:")
            cols = st.columns(k)
            for i, (col, neighbor_idx) in enumerate(zip(cols, indices[0])):
                with col:
                    neighbor_img = x_train[neighbor_idx]
                    neighbor_class = class_names[y_train_flat[neighbor_idx]]
                    # 确保图像数据在正确的范围内
                    if neighbor_img.max() > 1.0:
                        neighbor_img = neighbor_img / 255.0
                    st.image(neighbor_img, width=80, clamp=True)
                    st.write(neighbor_class)

# 自定义 SVM Loss
class SVMLoss(nn.Module):
    def __init__(self):
        super(SVMLoss, self).__init__()
    
    def forward(self, outputs, targets):
        batch_size = outputs.shape[0]
        correct_scores = outputs[range(batch_size), targets].unsqueeze(1)
        margins = torch.clamp(outputs - correct_scores + 1, min=0)
        margins[range(batch_size), targets] = 0
        loss = margins.sum() / batch_size
        return loss

# 模块三：线性分类器 + 可视化
def linear_classifier():
    st.header("线性分类器 + 可视化 (CIFAR-10)")
    
    # 加载完整数据
    with st.spinner("加载 CIFAR-10 数据..."):
        x_train, y_train, x_test, y_test = load_cifar10_full()
    
    st.info("使用 CIFAR-10 数据集：5000 张训练图片，10000 张测试图片")
    
    # 数据预处理 - 先展平标签
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # 确保数据在正确的范围内
    if x_train.max() > 1.0:
        x_train = x_train.astype(np.float32) / 255.0
    if x_test.max() > 1.0:
        x_test = x_test.astype(np.float32) / 255.0
    
    # 先显示一些样本图片看看数据是否正常
    st.subheader("样本图片预览")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(x_train[i], width=80, clamp=True)
            st.write(f"类别: {class_names[y_train[i]]}")
    
    # 转换为 PyTorch 张量
    x_train_tensor = torch.from_numpy(x_train).reshape(-1, 3*32*32).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    x_test_tensor = torch.from_numpy(x_test).reshape(-1, 3*32*32).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    # 线性分类器模型
    class LinearClassifier(nn.Module):
        def __init__(self):
            super(LinearClassifier, self).__init__()
            self.linear = nn.Linear(3*32*32, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    # 3.1 线性分类器训练与模板可视化
    st.subheader("3.1 线性分类器训练与模板可视化")
    
    if 'model' not in st.session_state:
        st.session_state.model = LinearClassifier()
        st.session_state.trained = False
    
    if st.button("训练线性分类器"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_text = st.empty()
        
        model = st.session_state.model
        criterion = SVMLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        epochs = 100
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"训练进度: {epoch + 1}/{epochs}")
            loss_text.text(f"当前 Loss: {loss.item():.4f}")
        
        st.session_state.model = model
        st.session_state.losses = losses
        st.session_state.trained = True
        status_text.text("训练完成！")
        st.success("线性分类器训练完成！")
    elif not st.session_state.trained:
        st.write("点击'训练线性分类器'按钮开始训练")
    
    if st.session_state.trained:
        # 绘制损失曲线
        st.subheader("训练损失曲线")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(st.session_state.losses) + 1), st.session_state.losses)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('训练损失曲线')
        st.pyplot(fig)
    
    # 3.2 SGD vs SGD+Momentum 对比
    st.subheader("3.2 SGD vs SGD+Momentum 梯度下降对比")
    
    if st.button("运行优化器对比"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 重置模型
        model_sgd = LinearClassifier()
        model_momentum = LinearClassifier()
        
        # 复制初始权重
        model_momentum.load_state_dict(model_sgd.state_dict())
        
        criterion = SVMLoss()
        optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
        optimizer_momentum = optim.SGD(model_momentum.parameters(), lr=0.01, momentum=0.9)
        
        epochs = 20
        losses_sgd = []
        losses_momentum = []
        
        for epoch in range(epochs):
            # SGD
            optimizer_sgd.zero_grad()
            outputs_sgd = model_sgd(x_train_tensor.float())
            loss_sgd = criterion(outputs_sgd, y_train_tensor)
            loss_sgd.backward()
            optimizer_sgd.step()
            losses_sgd.append(loss_sgd.item())
            
            # SGD + Momentum
            optimizer_momentum.zero_grad()
            outputs_momentum = model_momentum(x_train_tensor.float())
            loss_momentum = criterion(outputs_momentum, y_train_tensor)
            loss_momentum.backward()
            optimizer_momentum.step()
            losses_momentum.append(loss_momentum.item())
            
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"训练进度: {epoch + 1}/{epochs}")
        
        status_text.text("训练完成！")
        st.success("优化器对比训练完成！")
        
        # 绘制对比曲线
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, epochs + 1), losses_sgd, label='SGD')
        ax.plot(range(1, epochs + 1), losses_momentum, label='SGD+Momentum')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('SGD vs SGD+Momentum 损失对比')
        ax.legend()
        st.pyplot(fig)
    
    # 3.3 不同损失函数的计算过程演示
    st.subheader("3.3 不同损失函数的计算过程演示")
    
    # 用户输入
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_score = st.number_input("Cat 分数", value=3.2)
    with col2:
        car_score = st.number_input("Car 分数", value=5.1)
    with col3:
        frog_score = st.number_input("Frog 分数", value=-1.7)
    
    correct_class = st.selectbox("正确类别", ['cat', 'car', 'frog'])
    
    # 计算 SVM Loss
    if st.button("计算 SVM Loss"):
        scores = np.array([cat_score, car_score, frog_score])
        class_idx = {'cat': 0, 'car': 1, 'frog': 2}[correct_class]
        correct_score = scores[class_idx]
        
        st.write("### SVM Loss 计算过程")
        st.write(f"正确类别: {correct_class} (分数: {correct_score})")
        st.write(f"所有分数: Cat={cat_score}, Car={car_score}, Frog={frog_score}")
        
        margins = []
        for i, cls in enumerate(['cat', 'car', 'frog']):
            if i != class_idx:
                margin = scores[i] - correct_score + 1
                margins.append(margin)
                st.write(f"{cls}: margin = {scores[i]} - {correct_score} + 1 = {margin:.2f}")
        
        max_margins = [max(0, m) for m in margins]
        for i, (cls, m) in enumerate(zip(['cat', 'car', 'frog'], max_margins)):
            if i != class_idx:
                st.write(f"{cls}: max(0, margin) = {m:.2f}")
        
        svm_loss = sum(max_margins)
        st.write(f"最终 SVM Loss: {svm_loss:.2f}")
    
    # 计算 Cross-Entropy Loss
    if st.button("计算 Cross-Entropy Loss"):
        scores = np.array([cat_score, car_score, frog_score])
        class_idx = {'cat': 0, 'car': 1, 'frog': 2}[correct_class]
        
        st.write("### Cross-Entropy Loss 计算过程")
        st.write(f"正确类别: {correct_class}")
        st.write(f"所有分数: Cat={cat_score}, Car={car_score}, Frog={frog_score}")
        
        exp_scores = np.exp(scores)
        st.write(f"指数分数: Cat={exp_scores[0]:.2f}, Car={exp_scores[1]:.2f}, Frog={exp_scores[2]:.2f}")
        
        sum_exp = np.sum(exp_scores)
        st.write(f"指数分数之和: {sum_exp:.2f}")
        
        softmax = exp_scores / sum_exp
        st.write(f"Softmax: Cat={softmax[0]:.4f}, Car={softmax[1]:.4f}, Frog={softmax[2]:.4f}")
        
        ce_loss = -np.log(softmax[class_idx])
        st.write(f"最终 Cross-Entropy Loss: {ce_loss:.4f}")

# 主应用
st.title("机器学习交互式 Web 应用")

# 侧边栏数据说明
st.sidebar.subheader("数据集说明")
st.sidebar.info("""
**最小二乘线性回归**: 模拟生成的线性数据（带噪声）

**KNN 分类器**: CIFAR-10 数据集（1000训练 + 200测试）

**线性分类器**: CIFAR-10 数据集（5000训练 + 10000测试）
""")

# 使用 tabs 切换模块
tab1, tab2, tab3 = st.tabs(["最小二乘线性回归", "KNN 分类器", "线性分类器 + 可视化"])

with tab1:
    linear_regression_demo()

with tab2:
    knn_classifier()

with tab3:
    linear_classifier()
