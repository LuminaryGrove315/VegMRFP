# ply_io.py
import open3d as o3d
import numpy as np
from config import PLY_XYZ_ATTR, PLY_SH_ATTRS, PLY_OUTPUT_LABEL_ATTR

def load_gaussian_ply(ply_path: str):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts_num = len(pcd.points)
    # 读取三维坐标
    xyz = np.asarray(pcd.points, dtype=np.float32)

    # 读取自定义SH特征（open3d自定义属性存在 pcd.point_data）
    sh_feature = np.zeros((pts_num, len(PLY_SH_ATTRS)), dtype=np.float32)
    for idx, attr_name in enumerate(PLY_SH_ATTRS):
        sh_feature[:, idx] = np.asarray(pcd.point_data[attr_name], dtype=np.float32)

    print(f"成功加载PLY图元：总数 {pts_num} 个")
    print(f"坐标shape: {xyz.shape}, SH特征shape: {sh_feature.shape}")
    return xyz, sh_feature

def save_seg_ply(ply_save_path: str, xyz: np.ndarray, sh_mat: np.ndarray, cl_label: np.ndarray):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # 写入原始SH特征
    for idx, attr_name in enumerate(PLY_SH_ATTRS):
        pcd.point_data[attr_name] = sh_mat[:, idx]

    # 写入枝冠分类标签
    pcd.point_data[PLY_OUTPUT_LABEL_ATTR] = cl_label.astype(np.float32)

    # 保存PLY文件
    o3d.io.write_point_cloud(ply_save_path, pcd, write_ascii=False)
    print(f"分割结果已保存至：{ply_save_path}")
    print(f"枝干(cl=0): {np.sum(cl_label==0)} | 树冠(cl=1): {np.sum(cl_label==1)} | 噪声: {np.sum(cl_label==-1)}")