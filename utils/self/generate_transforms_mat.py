import os
import json
import numpy as np
import scipy.io


def w2c_to_c2w_opengl(R, T):
    """
    将 OpenCV 的 World-to-Camera (W2C) 转换为 OpenGL 的 Camera-to-World (C2W)
    """
    # 构造 4x4 的 W2C 矩阵
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.flatten()

    # 1. 求逆得到 C2W
    c2w = np.linalg.inv(w2c)

    # 2. 翻转 Y 和 Z 轴适配 OpenGL (2DGS/NeRF 标准)
    c2w[0:3, 1] *= -1.0
    c2w[0:3, 2] *= -1.0

    return c2w


def generate_transforms_for_all_objects(dataset_root, output_root):
    # DiLiGenT-MV 的 5 个物体
    objects = ['bear', 'buddha', 'cow', 'pot2', 'reading']

    for obj_name in objects:
        obj_in_dir = os.path.join(dataset_root, obj_name + 'PNG')
        calib_mat_path = os.path.join(obj_in_dir, 'Calib_Results.mat')

        if not os.path.exists(calib_mat_path):
            print(f"⚠️ 在 {obj_in_dir} 下未找到 Calib_Results.mat，跳过 {obj_name}...")
            continue

        print(f"正在处理 {obj_name}，读取标定文件: {calib_mat_path}")

        # 1. 加载 MATLAB 格式的标定文件
        mat_data = scipy.io.loadmat(calib_mat_path)

        # 2. 提取内参 K
        if 'KK' in mat_data:
            K = mat_data['KK']
        elif 'K' in mat_data:
            K = mat_data['K']
        else:
            print(f"⚠️ {obj_name} 的 MAT 文件中找不到内参矩阵 (KK/K)，跳过...")
            continue

        fl_x = float(K[0, 0])
        fl_y = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        # 原始图像大小
        w, h = 612, 512
        # 计算 FOV_X
        camera_angle_x = 2.0 * float(np.arctan(w / (2.0 * fl_x)))

        # 准备输出目录
        obj_out_dir = os.path.join(output_root, obj_name)
        os.makedirs(obj_out_dir, exist_ok=True)

        # 初始化 JSON 字典
        transforms_dict = {
            "camera_angle_x": camera_angle_x,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "frames": []
        }

        # 3. 提取 20 个视角的外参
        for i in range(1, 21):
            view_folder = f"view_{i:02d}"
            R_key = f'Rc_{i}'
            T_key = f'Tc_{i}'

            if R_key not in mat_data or T_key not in mat_data:
                print(f"  -> 找不到视角 {i} 的外参 ({R_key}, {T_key})，跳过该视角。")
                continue

            R = mat_data[R_key]
            T = mat_data[T_key]

            # 进行坐标系和矩阵转换
            c2w = w2c_to_c2w_opengl(R, T)

            # 加入 frames 列表 (注意这里的路径要和你生成的平均光照图对应)
            frame_info = {
                "file_path": f"images/{view_folder}.png",
                "transform_matrix": c2w.tolist()
            }
            transforms_dict["frames"].append(frame_info)

        # 4. 写入 JSON
        json_out_path = os.path.join(obj_out_dir, "transforms.json")
        with open(json_out_path, 'w') as f:
            json.dump(transforms_dict, f, indent=4)

        print(f"✅ [{obj_name}] 已成功生成 2DGS 专属位姿文件!")


if __name__ == "__main__":
    # --- 请配置你的实际路径 ---
    DILIGENT_MV_ROOT = "/data/sps/datasets/DiLiGenT-MV/DiLiGenT-MV/mvpmsData"
    OUTPUT_ROOT = "/data/sps/datasets/DiLiGenT-MV/DiLiGenT-MV-Averaged"
    # --------------------------

    generate_transforms_for_all_objects(DILIGENT_MV_ROOT, OUTPUT_ROOT)
    print("\n🎉 全部位姿提取完毕！")