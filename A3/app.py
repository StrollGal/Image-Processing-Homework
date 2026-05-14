import streamlit as st
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
import traceback

_font_candidates = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
_available = [f.name for f in fm.fontManager.ttflist]
matplotlib.rcParams['font.sans-serif'] = [f for f in _font_candidates if f in _available] or ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

MAX_DIM = 800

def resize_image(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def laplacian_pyramid_blend(img1, img2, mask1, mask2, levels=4):
    overlap = mask1 * mask2
    if overlap.sum() == 0:
        result = img2.copy()
        result[mask1 > 0] = img1[mask1 > 0]
        return result

    # 二值掩码：非重叠区取对应图像，重叠区从中间硬切
    alpha = np.zeros_like(mask1)
    alpha[(mask1 > 0) & (mask2 == 0)] = 1.0
    overlap_cols = np.where(overlap.any(axis=0))[0]
    if len(overlap_cols) > 1:
        mid = (overlap_cols[0] + overlap_cols[-1]) // 2
        alpha[:, overlap_cols[0]:mid+1] = 1.0

    h, w = img1.shape[:2]
    levels = min(levels, int(np.log2(min(h, w))) - 1)
    levels = max(1, levels)

    gp1, gp2, gpM = [img1], [img2], [alpha[..., None]]
    for _ in range(levels):
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))
        m_down = cv2.pyrDown(gpM[-1])
        gpM.append(m_down[..., None] if m_down.ndim == 2 else m_down)

    lp1, lp2 = [], []
    for i in range(levels):
        lp1.append(gp1[i] - cv2.pyrUp(gp1[i+1], dstsize=(gp1[i].shape[1], gp1[i].shape[0])))
        lp2.append(gp2[i] - cv2.pyrUp(gp2[i+1], dstsize=(gp2[i].shape[1], gp2[i].shape[0])))
    lp1.append(gp1[levels])
    lp2.append(gp2[levels])

    blended = [gpM[i] * lp1[i] + (1 - gpM[i]) * lp2[i] for i in range(levels + 1)]

    result = blended[levels]
    for i in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended[i].shape[1], blended[i].shape[0]))
        result += blended[i]

    return result

# 初始化会话状态
if 'state_img' not in st.session_state:
    st.session_state.state_img = None
if 'state_gray' not in st.session_state:
    st.session_state.state_gray = None
if 'state_gradient_magnitude' not in st.session_state:
    st.session_state.state_gradient_magnitude = None
if 'state_edges' not in st.session_state:
    st.session_state.state_edges = None
if 'state_harris_img' not in st.session_state:
    st.session_state.state_harris_img = None
if 'state_sift_img' not in st.session_state:
    st.session_state.state_sift_img = None
if 'state_img1' not in st.session_state:
    st.session_state.state_img1 = None
if 'state_img2' not in st.session_state:
    st.session_state.state_img2 = None
if 'state_gray1' not in st.session_state:
    st.session_state.state_gray1 = None
if 'state_gray2' not in st.session_state:
    st.session_state.state_gray2 = None
if 'state_img1_with_kp' not in st.session_state:
    st.session_state.state_img1_with_kp = None
if 'state_img2_with_kp' not in st.session_state:
    st.session_state.state_img2_with_kp = None
if 'state_img_initial_match' not in st.session_state:
    st.session_state.state_img_initial_match = None
if 'state_img_ransac_match' not in st.session_state:
    st.session_state.state_img_ransac_match = None
if 'state_result' not in st.session_state:
    st.session_state.state_result = None
if 'state_images' not in st.session_state:
    st.session_state.state_images = None
if 'state_panorama_simple' not in st.session_state:
    st.session_state.state_panorama_simple = None
if 'state_panorama_alpha' not in st.session_state:
    st.session_state.state_panorama_alpha = None
if 'state_panorama_pyramid' not in st.session_state:
    st.session_state.state_panorama_pyramid = None
if 'state_uploaded1' not in st.session_state:
    st.session_state.state_uploaded1 = False
if 'state_uploaded2' not in st.session_state:
    st.session_state.state_uploaded2 = False
if 'state_uploaded3' not in st.session_state:
    st.session_state.state_uploaded3 = False

# 设置页面标题
st.title('图像处理应用')

# 功能1：上传图片
st.header('1. 上传图片')
uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])
upload_button = st.button("上传图片")

if uploaded_file is not None and upload_button:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        img_array = resize_image(img_array)
        st.session_state.state_img = img_array
        st.session_state.state_uploaded1 = True

        st.session_state.state_gray = cv2.cvtColor(st.session_state.state_img, cv2.COLOR_RGB2GRAY)

        # Canny边缘检测
        st.session_state.state_edges = cv2.Canny(st.session_state.state_gray, 50, 150)

        # 梯度幅值
        sobel_x = cv2.Sobel(st.session_state.state_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(st.session_state.state_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(sobel_x**2 + sobel_y**2)
        st.session_state.state_gradient_magnitude = np.uint8(grad)

        # Harris角点检测
        harris_bgr = cv2.cvtColor(st.session_state.state_img, cv2.COLOR_RGB2BGR)
        harris_corners = cv2.cornerHarris(st.session_state.state_gray, 2, 3, 0.04)
        harris_corners = cv2.dilate(harris_corners, None)
        harris_bgr[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
        st.session_state.state_harris_img = cv2.cvtColor(harris_bgr, cv2.COLOR_BGR2RGB)

        # SIFT特征点检测
        sift_bgr = cv2.cvtColor(st.session_state.state_img, cv2.COLOR_RGB2BGR)
        sift = cv2.SIFT_create()
        keypoints, _ = sift.detectAndCompute(st.session_state.state_gray, None)
        sift_bgr = cv2.drawKeypoints(sift_bgr, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        st.session_state.state_sift_img = cv2.cvtColor(sift_bgr, cv2.COLOR_BGR2RGB)

        st.success("图片处理完成！")
    except Exception as e:
        st.error(f"处理失败: {str(e)}")
        st.code(traceback.format_exc())

# 显示功能1-3的结果
if st.session_state.state_uploaded1 and st.session_state.state_img is not None:
    st.subheader('原始图像')
    st.image(st.session_state.state_img, use_container_width=True)

    if st.session_state.state_gray is not None:
        # 功能2：边缘检测（Canny方法）
        st.header('2. 边缘检测（Canny方法）')
        low_threshold = st.slider('低阈值', 0, 255, 50, key="canny_low")
        high_threshold = st.slider('高阈值', 0, 255, 150, key="canny_high")

        st.session_state.state_edges = cv2.Canny(st.session_state.state_gray, low_threshold, high_threshold)

        st.subheader('边缘检测对比')
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.state_gradient_magnitude, use_container_width=True, clamp=True)
            st.caption('梯度幅值（非最大值抑制前）')
        with col2:
            st.image(st.session_state.state_edges, use_container_width=True, clamp=True)
            st.caption('Canny边缘检测（非最大值抑制后）')

        # 功能3：特征点检测
        st.header('3. 特征点检测')
        st.subheader('特征点检测对比')
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.state_harris_img, use_container_width=True)
            st.caption('Harris角点检测（红色点）')
        with col2:
            st.image(st.session_state.state_sift_img, use_container_width=True)
            st.caption('SIFT特征点检测（绿色圆圈）')
else:
    st.header('2. 边缘检测（Canny方法）')
    st.header('3. 特征点检测')

# 功能4：上传一对具有重叠区域的图像
st.header('4. 上传一对具有重叠区域的图像')
uploaded_file1 = st.file_uploader("选择第一张图片", type=["jpg", "jpeg", "png"], key="img1")
uploaded_file2 = st.file_uploader("选择第二张图片", type=["jpg", "jpeg", "png"], key="img2")
upload_button2 = st.button("上传两张图片")

if uploaded_file1 is not None and uploaded_file2 is not None and upload_button2:
    try:
        image1 = Image.open(uploaded_file1).convert('RGB')
        image2 = Image.open(uploaded_file2).convert('RGB')
        st.session_state.state_img1 = resize_image(np.array(image1))
        st.session_state.state_img2 = resize_image(np.array(image2))
        st.session_state.state_uploaded2 = True

        st.session_state.state_gray1 = cv2.cvtColor(st.session_state.state_img1, cv2.COLOR_RGB2GRAY)
        st.session_state.state_gray2 = cv2.cvtColor(st.session_state.state_img2, cv2.COLOR_RGB2GRAY)

        # 特征点检测
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(st.session_state.state_gray1, None)
        kp2, des2 = sift.detectAndCompute(st.session_state.state_gray2, None)

        # 可视化特征点
        img1_bgr = cv2.cvtColor(st.session_state.state_img1, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(st.session_state.state_img2, cv2.COLOR_RGB2BGR)
        st.session_state.state_img1_with_kp = cv2.cvtColor(
            cv2.drawKeypoints(img1_bgr, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),
            cv2.COLOR_BGR2RGB)
        st.session_state.state_img2_with_kp = cv2.cvtColor(
            cv2.drawKeypoints(img2_bgr, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),
            cv2.COLOR_BGR2RGB)

        # 特征点匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # 可视化初始匹配
        st.session_state.state_img_initial_match = cv2.cvtColor(
            cv2.drawMatches(img1_bgr, kp1, img2_bgr, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS),
            cv2.COLOR_BGR2RGB)

        # RANSAC
        if len(good_matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                matches_mask = mask.ravel().tolist()

                draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                st.session_state.state_img_ransac_match = cv2.cvtColor(
                    cv2.drawMatches(img1_bgr, kp1, img2_bgr, kp2, good_matches, None, **draw_params),
                    cv2.COLOR_BGR2RGB)

                # 变换和对齐
                h, w = st.session_state.state_img1.shape[:2]
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                x_min = min(0, np.min(dst[:, :, 0]))
                x_max = max(st.session_state.state_img2.shape[1], np.max(dst[:, :, 0]))
                y_min = min(0, np.min(dst[:, :, 1]))
                y_max = max(st.session_state.state_img2.shape[0], np.max(dst[:, :, 1]))

                translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

                warped = cv2.warpPerspective(st.session_state.state_img1, translation_matrix @ M, (int(x_max - x_min), int(y_max - y_min)))

                aligned = np.zeros_like(warped)
                y1, y2 = -int(y_min), st.session_state.state_img2.shape[0]-int(y_min)
                x1, x2 = -int(x_min), st.session_state.state_img2.shape[1]-int(x_min)
                if y2 <= warped.shape[0] and x2 <= warped.shape[1]:
                    aligned[y1:y2, x1:x2] = st.session_state.state_img2

                st.session_state.state_result = warped.copy()
                mask_arr = (aligned != 0).any(axis=2)
                st.session_state.state_result[mask_arr] = aligned[mask_arr]
            else:
                st.warning('无法计算单应性矩阵')
        else:
            st.warning('匹配点数量不足，无法进行RANSAC')

        st.success("图像匹配处理完成！")
    except Exception as e:
        st.error(f"处理失败: {str(e)}")
        st.code(traceback.format_exc())

# 显示功能4-5的结果
if st.session_state.state_uploaded2 and st.session_state.state_img1 is not None and st.session_state.state_img2 is not None:
    st.subheader('原始图像')
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.state_img1, use_container_width=True)
        st.caption('第一张图片')
    with col2:
        st.image(st.session_state.state_img2, use_container_width=True)
        st.caption('第二张图片')

    if st.session_state.state_gray1 is not None and st.session_state.state_gray2 is not None:
        st.header('5. 图像匹配流程')

        if st.session_state.state_img1_with_kp is not None:
            st.subheader('1. 特征点检测')
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.state_img1_with_kp, use_container_width=True)
                st.caption('第一张图片的SIFT特征点')
            with col2:
                st.image(st.session_state.state_img2_with_kp, use_container_width=True)
                st.caption('第二张图片的SIFT特征点')

            st.subheader('2. 初始匹配')
            st.image(st.session_state.state_img_initial_match, use_container_width=True)
            st.caption('基于SIFT的初始特征点匹配')

            if st.session_state.state_img_ransac_match is not None:
                st.subheader('3. RANSAC过滤')
                st.image(st.session_state.state_img_ransac_match, use_container_width=True)
                st.caption('RANSAC过滤后的匹配点')

                st.subheader('4. 变换和对齐')
                st.image(st.session_state.state_result, use_container_width=True)
                st.caption('图像对齐结果')
else:
    st.header('5. 图像匹配流程')

# 功能6：上传多幅有重叠区域的图像
st.header('6. 上传多幅有重叠区域的图像')
uploaded_files = st.file_uploader("选择多张图片（至少2张）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
upload_button3 = st.button("上传多张图片")

if uploaded_files is not None and len(uploaded_files) >= 2 and upload_button3:
    try:
        st.session_state.state_images = []
        for file in uploaded_files:
            image = Image.open(file).convert('RGB')
            st.session_state.state_images.append(resize_image(np.array(image)))
        st.session_state.state_uploaded3 = True

        def stitch_images(images, blending='simple'):
            orb = cv2.ORB_create()

            all_kp = []
            all_des = []
            all_images = images.copy()

            for img in all_images:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                kp, des = orb.detectAndCompute(gray, None)
                all_kp.append(kp)
                all_des.append(des)

            panorama = all_images[0]

            for i in range(1, len(all_images)):
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(all_des[0], all_des[i])
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) < 4:
                    continue

                src_pts = np.float32([all_kp[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([all_kp[i][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                if M is None:
                    continue

                h, w = all_images[i].shape[:2]
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                x_min = min(0, np.min(dst[:, :, 0]))
                x_max = max(panorama.shape[1], np.max(dst[:, :, 0]))
                y_min = min(0, np.min(dst[:, :, 1]))
                y_max = max(panorama.shape[0], np.max(dst[:, :, 1]))

                translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

                warped = cv2.warpPerspective(all_images[i], translation_matrix @ M, (int(x_max - x_min), int(y_max - y_min)))

                panorama_warped = np.zeros_like(warped)
                py1, py2 = -int(y_min), panorama.shape[0]-int(y_min)
                px1, px2 = -int(x_min), panorama.shape[1]-int(x_min)
                if py2 <= warped.shape[0] and px2 <= warped.shape[1]:
                    panorama_warped[py1:py2, px1:px2] = panorama

                if blending == '简单拼接':
                    result = warped.copy()
                    m = (panorama_warped != 0).any(axis=2)
                    result[m] = panorama_warped[m]
                elif blending == 'Alpha融合':
                    mask1 = (panorama_warped > 0).any(axis=2).astype(np.float32)
                    mask2 = (warped > 0).any(axis=2).astype(np.float32)
                    overlap = mask1 * mask2
                    if overlap.sum() == 0:
                        result = warped.copy()
                        result[mask1 > 0] = panorama_warped[mask1 > 0]
                    else:
                        alpha = np.zeros_like(mask1)
                        alpha[mask1 > 0] = 1.0
                        col_indices = np.where(overlap.any(axis=0))[0]
                        if len(col_indices) > 1:
                            left, right = col_indices[0], col_indices[-1]
                            gradient = np.linspace(1, 0, right - left + 1)
                            alpha[:, left:right+1] = np.tile(gradient, (alpha.shape[0], 1))
                            alpha *= mask1
                        result = (alpha[..., None] * panorama_warped.astype(np.float32) +
                                  (1 - alpha[..., None]) * warped.astype(np.float32))
                        result = np.clip(result, 0, 255).astype(np.uint8)
                elif blending == '金字塔融合':
                    mask1 = (panorama_warped > 0).any(axis=2).astype(np.float32)
                    mask2 = (warped > 0).any(axis=2).astype(np.float32)
                    result = laplacian_pyramid_blend(
                        panorama_warped.astype(np.float32),
                        warped.astype(np.float32), mask1, mask2)
                    result = np.clip(result, 0, 255).astype(np.uint8)

                panorama = result
                all_kp[0] = all_kp[i]
                all_des[0] = all_des[i]

            return panorama

        st.session_state.state_panorama_simple = stitch_images(st.session_state.state_images, '简单拼接')
        st.session_state.state_panorama_alpha = stitch_images(st.session_state.state_images, 'Alpha融合')
        st.session_state.state_panorama_pyramid = stitch_images(st.session_state.state_images, '金字塔融合')

        st.success("全景拼接完成！")
    except Exception as e:
        st.error(f"处理失败: {str(e)}")
        st.code(traceback.format_exc())

# 显示功能6-7的结果
if st.session_state.state_uploaded3 and st.session_state.state_images is not None:
    st.subheader('原始图像')
    cols = st.columns(len(st.session_state.state_images))
    for i, (col, img) in enumerate(zip(cols, st.session_state.state_images)):
        with col:
            st.image(img, use_container_width=True)
            st.caption(f'图片{i+1}')

    st.header('7. 全景拼接')

    st.subheader('不同blending方式对比')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(st.session_state.state_panorama_simple, use_container_width=True)
        st.caption('简单拼接')

    with col2:
        st.image(st.session_state.state_panorama_alpha, use_container_width=True)
        st.caption('Alpha融合')

    with col3:
        st.image(st.session_state.state_panorama_pyramid, use_container_width=True)
        st.caption('金字塔融合')
else:
    st.header('7. 全景拼接')
