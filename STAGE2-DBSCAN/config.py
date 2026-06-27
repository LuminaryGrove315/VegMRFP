K_NN = 16
SH_DIM = 48
# SH特征融合权重 w
W_SH = 0.4
# DBSCAN聚类参数
EPS = 0.05
MIN_PTS = 15000
# 2DGS结果
PLY_XYZ_ATTR = ["x", "y", "z"]
PLY_SH_ATTRS = [f"sh_{i}" for i in range(SH_DIM)]
PLY_OUTPUT_LABEL_ATTR = "cl"  