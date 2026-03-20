import os
import cv2
import numpy as np
from tqdm import tqdm


def generate_light_averaged_images(dataset_root, output_root):
    """
    读取 DiLiGenT-MV 数据集，进行光度归一化并生成多视角的光照平均图像

    :param dataset_root: DiLiGenT-MV 数据集根目录 (包含 bear, buddha 等文件夹)
    :param output_root: 自定义的输出路径
    """
    # DiLiGenT-MV 数据集中的 5 个物体
    objects = ['bear', 'buddha', 'cow', 'pot2', 'reading']

    for obj_name in objects:
        obj_in_dir = os.path.join(dataset_root, obj_name + 'PNG')
        if not os.path.exists(obj_in_dir):
            print(f"未找到物体文件夹: {obj_in_dir}，跳过...")
            continue

        # 为每个物体在输出目录下创建一个 images 文件夹
        obj_out_dir = os.path.join(output_root, obj_name, "images")
        os.makedirs(obj_out_dir, exist_ok=True)

        # 获取所有的视角文件夹 (view_01 到 view_20)
        view_folders = [f for f in os.listdir(obj_in_dir) if f.startswith('view_')]
        view_folders.sort()

        print(f"\n开始处理物体: {obj_name} ...")

        for view_folder in tqdm(view_folders, desc=f"Processing Views for {obj_name}"):
            view_dir = os.path.join(obj_in_dir, view_folder)

            # 1. 读取光照强度矩阵 (3x96 矩阵: 每行分别是 R, G, B 通道的强度)
            intensities_path = os.path.join(view_dir, 'light_intensities.txt')
            if not os.path.exists(intensities_path):
                continue
            # shape: (3, 96)
            light_intensities = np.loadtxt(intensities_path)

            # 2. 读取掩码文件
            mask_path = os.path.join(view_dir, 'mask.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # 将 mask 归一化为 0 和 1
                mask = (mask > 127).astype(np.float32)

            # 获取 96 张图像 (001.png 到 096.png)
            img_files = [f"{str(i).zfill(3)}.png" for i in range(1, 97)]

            sum_img = None
            valid_images_count = 0

            for idx, img_name in enumerate(img_files):
                img_path = os.path.join(view_dir, img_name)
                if not os.path.exists(img_path):
                    continue

                # 以 16-bit 原始深度读取图像 (OpenCV 默认读取为 BGR 格式)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                # 转为浮点型以进行除法计算
                img = img.astype(np.float32)

                # 转为 RGB 格式，方便与 txt 中的 RGB 强度对应
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 3. 核心步骤：光度归一化 (除以光照强度)
                # light_intensities 形状为 (96, 3)
                # [idx, 0] 是 R, [idx, 1] 是 G, [idx, 2] 是 B
                r_int = light_intensities[idx, 0]
                g_int = light_intensities[idx, 1]
                b_int = light_intensities[idx, 2]

                # 避免除以 0
                r_int = r_int if r_int > 0 else 1.0
                g_int = g_int if g_int > 0 else 1.0
                b_int = b_int if b_int > 0 else 1.0

                img[:, :, 0] /= r_int
                img[:, :, 1] /= g_int
                img[:, :, 2] /= b_int

                # 累加图像
                if sum_img is None:
                    sum_img = np.zeros_like(img, dtype=np.float32)
                sum_img += img
                valid_images_count += 1

            if valid_images_count == 0:
                continue

            # 4. 计算平均值
            avg_img = sum_img / valid_images_count

            # 5. 应用 Mask (将背景置黑)，去除噪点
            if mask is not None:
                # mask 拓展为 3 通道
                mask_3d = np.expand_dims(mask, axis=-1)
                avg_img = avg_img * mask_3d

            # 6. 数据类型转换与保存
            # 因为原始图像是 16-bit (最大值 65535)，我们需要将其映射为 2DGS 适用的标准的 8-bit (0-255) RGB 图像
            # 找到当前平均图像的最大值，进行自适应的全局亮度映射 (或者直接除以 65535.0 * 255)
            # 论文通常建议通过简单的线性缩放，这里为了视觉效果更佳，进行 99% 的截断归一化
            max_val = np.percentile(avg_img[mask == 1] if mask is not None else avg_img, 99.9)
            avg_img = (avg_img / max_val) * 255.0

            avg_img_uint8 = np.clip(avg_img, 0, 255).astype(np.uint8)

            # 存回 BGR 格式给 cv2 写入
            avg_img_uint8 = cv2.cvtColor(avg_img_uint8, cv2.COLOR_RGB2BGR)

            # 最终保存的文件名例如 view_01.png
            out_name = f"{view_folder}.png"
            out_path = os.path.join(obj_out_dir, out_name)

            cv2.imwrite(out_path, avg_img_uint8)


if __name__ == "__main__":
    # --- 请在这里配置你的路径 ---
    # DiLiGenT-MV 数据集根目录 (即包含 bear, buddha 等文件夹的目录)
    DILIGENT_MV_ROOT = "/data/sps/datasets/DiLiGenT-MV/DiLiGenT-MV/mvpmsData"

    # 你希望保存生成结果的自定义路径
    OUTPUT_ROOT = "/data/sps/datasets/DiLiGenT-MV/DiLiGenT-MV-Averaged"
    # --------------------------

    print("启动 DiLiGenT-MV 数据集处理脚本...")
    generate_light_averaged_images(DILIGENT_MV_ROOT, OUTPUT_ROOT)
    print(f"\n所有光照平均图像生成完毕！保存在: {OUTPUT_ROOT}")