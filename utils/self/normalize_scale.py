import json
import numpy as np
import os


def normalize_json_scale():
    # 我们希望相机的平均距离缩放到 3.0 (2DGS/NeRF 最舒服的尺度)
    TARGET_RADIUS = 3.0

    # 指向你的 bear 目录下的两个 json 文件
    work_dir = "/data/sps/datasets/DiLiGenT-MV/DiLiGenT-MV-Averaged/bear"
    files = ["transforms_train.json", "transforms_test.json"]

    for file_name in files:
        file_path = os.path.join(work_dir, file_name)
        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        # 1. 计算所有相机到原点的平均距离
        distances = []
        for frame in data["frames"]:
            c2w = np.array(frame["transform_matrix"])
            dist = np.linalg.norm(c2w[:3, 3])
            distances.append(dist)

        avg_dist = np.mean(distances)
        scale_factor = TARGET_RADIUS / avg_dist

        print(f"处理 [{file_name}]:")
        print(f"  -> 原相机平均距离: {avg_dist:.2f}")
        print(f"  -> 缩放比例 (Scale Factor): {scale_factor:.6f}")

        # 2. 对所有相机的平移向量应用缩放
        for frame in data["frames"]:
            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 3] *= scale_factor
            frame["transform_matrix"] = c2w.tolist()

        # 3. 覆写保存
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    print("✅ 位姿尺度归一化完成！初始随机点现在能完美覆盖物体了。")


if __name__ == "__main__":
    normalize_json_scale()