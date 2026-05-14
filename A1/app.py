from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import os

class ImageProcessingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("图像处理工具")
        self.root.geometry("1200x800")
        
        # 全局变量
        self.original_image = None
        self.image_path = None
        self.resized_original = None  # 保存调整大小后的原始图像
        
        # 创建主滚动框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建垂直滚动条
        self.v_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建滚动画布
        self.canvas = tk.Canvas(self.main_frame, yscrollcommand=self.v_scrollbar.set)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 配置滚动条
        self.v_scrollbar.config(command=self.canvas.yview)
        
        # 创建内部框架
        self.inner_frame = ttk.Frame(self.canvas, padding="10")
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor=tk.NW)
        
        # 绑定事件以更新滚动区域
        self.inner_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        # 创建上传区域
        self.upload_frame = ttk.LabelFrame(self.inner_frame, text="上传图片", padding="10")
        self.upload_frame.pack(fill=tk.X, pady=10)
        
        self.upload_button = ttk.Button(self.upload_frame, text="选择图片", command=self.upload_image)
        self.upload_button.pack()
        
        self.image_label = ttk.Label(self.upload_frame)
        self.image_label.pack(pady=10)
        
        # 创建颜色空间转换区域
        self.color_frame = ttk.LabelFrame(self.inner_frame, text="颜色空间转换", padding="10")
        self.color_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建图像操作和对比区域
        self.operation_frame = ttk.Frame(self.inner_frame)
        self.operation_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 左侧：图像操作
        self.control_frame = ttk.LabelFrame(self.operation_frame, text="图像操作", padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # 缩放控制
        ttk.Label(self.control_frame, text="缩放比例:").pack(anchor=tk.W, pady=5)
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_slider = ttk.Scale(self.control_frame, from_=0.1, to=3.0, orient=tk.HORIZONTAL, variable=self.scale_var, command=self.update_image)
        self.scale_slider.pack(fill=tk.X, pady=5)
        self.scale_label = ttk.Label(self.control_frame, text="1.0")
        self.scale_label.pack(anchor=tk.W)
        
        # 旋转控制
        ttk.Label(self.control_frame, text="旋转角度:").pack(anchor=tk.W, pady=5)
        self.rotate_var = tk.IntVar(value=0)
        self.rotate_slider = ttk.Scale(self.control_frame, from_=0, to=360, orient=tk.HORIZONTAL, variable=self.rotate_var, command=self.update_image)
        self.rotate_slider.pack(fill=tk.X, pady=5)
        self.rotate_label = ttk.Label(self.control_frame, text="0°")
        self.rotate_label.pack(anchor=tk.W)
        
        # 右侧：插值算法对比
        self.comparison_frame = ttk.LabelFrame(self.operation_frame, text="插值算法对比", padding="10")
        self.comparison_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        self.nearest_label = ttk.Label(self.comparison_frame, text="最近邻插值")
        self.nearest_label.pack()
        self.nearest_image_label = ttk.Label(self.comparison_frame)
        self.nearest_image_label.pack(pady=10)
        
        self.bilinear_label = ttk.Label(self.comparison_frame, text="双线性插值")
        self.bilinear_label.pack()
        self.bilinear_image_label = ttk.Label(self.comparison_frame)
        self.bilinear_image_label.pack(pady=10)
    
    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.original_image = Image.open(self.image_path)
            # 调整图像大小以适应窗口
            width, height = self.original_image.size
            max_size = 400
            if width > max_size or height > max_size:
                ratio = min(max_size / width, max_size / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                resized_image = self.original_image.resize((new_width, new_height))
            else:
                resized_image = self.original_image
            
            # 保存调整大小后的原始图像
            self.resized_original = resized_image
            
            # 显示原始图像
            self.display_image(self.image_label, resized_image)
            
            # 更新颜色空间转换结果
            self.update_color_space_results()
            
            # 更新插值算法对比
            self.update_image()
    
    def display_image(self, label, image):
        # 保存临时图像并显示
        temp_path = "temp_image.png"
        image.save(temp_path)
        img = tk.PhotoImage(file=temp_path)
        label.config(image=img)
        label.image = img  # 保持引用
        os.remove(temp_path)
    
    def update_color_space_results(self):
        if not self.original_image:
            return
        
        # 清空颜色空间转换区域
        for widget in self.color_frame.winfo_children():
            widget.destroy()
        
        # 调整图像大小以适应窗口
        width, height = self.original_image.size
        max_size = 200
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            resized_original = self.original_image.resize((new_width, new_height))
        else:
            resized_original = self.original_image
        
        # 生成RGB通道
        img_array = np.array(resized_original)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # 红色通道
            red_channel = np.zeros_like(img_array)
            red_channel[:, :, 0] = img_array[:, :, 0]
            red_image = Image.fromarray(red_channel)
            
            # 绿色通道
            green_channel = np.zeros_like(img_array)
            green_channel[:, :, 1] = img_array[:, :, 1]
            green_image = Image.fromarray(green_channel)
            
            # 蓝色通道
            blue_channel = np.zeros_like(img_array)
            blue_channel[:, :, 2] = img_array[:, :, 2]
            blue_image = Image.fromarray(blue_channel)
            
            # 创建网格布局
            frame1 = ttk.Frame(self.color_frame)
            frame1.pack(fill=tk.X, pady=5)
            
            red_label = ttk.Label(frame1, text="RGB - 红色通道")
            red_label.pack(side=tk.LEFT, padx=10)
            red_img_label = ttk.Label(frame1)
            red_img_label.pack(side=tk.LEFT, padx=10)
            self.display_image(red_img_label, red_image)
            
            green_label = ttk.Label(frame1, text="RGB - 绿色通道")
            green_label.pack(side=tk.LEFT, padx=10)
            green_img_label = ttk.Label(frame1)
            green_img_label.pack(side=tk.LEFT, padx=10)
            self.display_image(green_img_label, green_image)
            
            blue_label = ttk.Label(frame1, text="RGB - 蓝色通道")
            blue_label.pack(side=tk.LEFT, padx=10)
            blue_img_label = ttk.Label(frame1)
            blue_img_label.pack(side=tk.LEFT, padx=10)
            self.display_image(blue_img_label, blue_image)
            
            # 生成HSV通道
            hsv_image = resized_original.convert("HSV")
            hsv_array = np.array(hsv_image)
            
            # 色相通道 - 保持饱和度和亮度为最大值
            h_channel = np.zeros_like(hsv_array)
            h_channel[:, :, 0] = hsv_array[:, :, 0]  # 色相
            h_channel[:, :, 1] = 255  # 最大饱和度
            h_channel[:, :, 2] = 255  # 最大亮度
            h_image = Image.fromarray(h_channel, "HSV").convert("RGB")
            
            # 饱和度通道 - 保持色相和亮度为固定值
            s_channel = np.zeros_like(hsv_array)
            s_channel[:, :, 0] = 0  # 固定色相为0（红色）
            s_channel[:, :, 1] = hsv_array[:, :, 1]  # 饱和度
            s_channel[:, :, 2] = 255  # 最大亮度
            s_image = Image.fromarray(s_channel, "HSV").convert("RGB")
            
            # 亮度通道 - 保持色相和饱和度为固定值
            v_channel = np.zeros_like(hsv_array)
            v_channel[:, :, 0] = 0  # 固定色相为0（红色）
            v_channel[:, :, 1] = 0  # 最小饱和度
            v_channel[:, :, 2] = hsv_array[:, :, 2]  # 亮度
            v_image = Image.fromarray(v_channel, "HSV").convert("RGB")
            
            frame2 = ttk.Frame(self.color_frame)
            frame2.pack(fill=tk.X, pady=5)
            
            h_label = ttk.Label(frame2, text="HSV - 色相通道")
            h_label.pack(side=tk.LEFT, padx=10)
            h_img_label = ttk.Label(frame2)
            h_img_label.pack(side=tk.LEFT, padx=10)
            self.display_image(h_img_label, h_image)
            
            s_label = ttk.Label(frame2, text="HSV - 饱和度通道")
            s_label.pack(side=tk.LEFT, padx=10)
            s_img_label = ttk.Label(frame2)
            s_img_label.pack(side=tk.LEFT, padx=10)
            self.display_image(s_img_label, s_image)
            
            v_label = ttk.Label(frame2, text="HSV - 亮度通道")
            v_label.pack(side=tk.LEFT, padx=10)
            v_img_label = ttk.Label(frame2)
            v_img_label.pack(side=tk.LEFT, padx=10)
            self.display_image(v_img_label, v_image)
    
    def update_image(self, *args):
        if not self.resized_original:
            return
        
        # 更新显示值
        scale = self.scale_var.get()
        rotation = self.rotate_var.get()
        self.scale_label.config(text=f"{scale:.1f}")
        self.rotate_label.config(text=f"{rotation}°")
        
        # 应用变换
        width, height = self.resized_original.size
        
        # 最近邻插值
        nearest_img = self.resized_original.rotate(rotation, expand=True, resample=Image.NEAREST)
        new_width = int(nearest_img.width * scale)
        new_height = int(nearest_img.height * scale)
        nearest_resized = nearest_img.resize((new_width, new_height), Image.NEAREST)
        self.display_image(self.nearest_image_label, nearest_resized)
        
        # 双线性插值
        bilinear_img = self.resized_original.rotate(rotation, expand=True, resample=Image.BILINEAR)
        bilinear_resized = bilinear_img.resize((new_width, new_height), Image.BILINEAR)
        self.display_image(self.bilinear_image_label, bilinear_resized)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingTool(root)
    root.mainloop()