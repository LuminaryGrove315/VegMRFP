# main_pipeline.py
from ply_io import load_gaussian_ply, save_seg_ply
from sld_cuda_wrapper import compute_sld_cuda
from dbscan_cluster import dbscan_sh_sld, map_crown_branch
import os

if __name__ == "__main__":
    # 输入输出PLY路径
    input_ply_path = "./data/input_plant.ply"
    output_ply_path = "./data/output_seg_plant.ply"

    # 检查输入文件
    if not os.path.exists(input_ply_path):
        raise FileNotFoundError(f"输入PLY不存在：{input_ply_path}")

    print("==============================================")
    
    print("\n[1] 加载PLY高斯图元数据...")
    xyz, sh_feature = load_gaussian_ply(input_ply_path)

   
    print("\n[2] GPU计算空间局部密度SLD...")
    sld_norm = compute_sld_cuda(xyz)

    print("\n[3] 执行SH+SLD融合DBSCAN密度聚类...")
    cluster_res = dbscan_sh_sld(sh_feature, sld_norm)


    print("\n[4] 根据SLD密度区分枝干、树冠簇...")
    seg_label = map_crown_branch(cluster_res, sld_norm)


    print("\n[5] 输出分割PLY文件...")
    save_seg_ply(output_ply_path, xyz, sh_feature, seg_label)
    print("\n FINISH！")