# sld_cuda_wrapper.py
import ctypes
import numpy as np
import pycuda.driver as cuda
from config import K_NN, THREADS_PER_BLOCK, SH_DIM


lib = ctypes.CDLL("./libcuda_core.so")

def compute_sld_cuda(xyz_np: np.ndarray) -> np.ndarray:
    n = xyz_np.shape[0]
    xyz_flat = xyz_np.astype(np.float32).reshape(-1)
    sld_host = np.zeros(n, dtype=np.float32)

   
    xyz_dev = cuda.mem_alloc(xyz_flat.nbytes)
    sld_dev = cuda.mem_alloc(sld_host.nbytes)
    cuda.memcpy_htod(xyz_dev, xyz_flat)

    # 调用CUDA核函数
    lib.launch_sld_kernel(
        ctypes.c_void_p(int(xyz_dev)),
        ctypes.c_void_p(int(sld_dev)),
        ctypes.c_int(n),
        ctypes.c_int(K_NN),
        ctypes.c_int(THREADS_PER_BLOCK)
    )


    cuda.memcpy_dtoh(sld_host, sld_dev)
    xyz_dev.free()
    sld_dev.free()

    # 全局归一化SLD
    sld_min = sld_host.min()
    sld_max = sld_host.max()
    sld_norm = (sld_host - sld_min) / (sld_max - sld_min + 1e-8)
    return sld_norm

def get_epsilon_neighbor_cuda(sh_mat: np.ndarray, sld_arr: np.ndarray, target_idx: int, eps: float, w: float):
    n = sh_mat.shape[0]
    sh_flat = sh_mat.astype(np.float32).reshape(-1)

    sh_dev = cuda.mem_alloc(sh_flat.nbytes)
    cuda.memcpy_htod(sh_dev, sh_flat)

    sld_dev = cuda.mem_alloc(sld_arr.nbytes)
    cuda.memcpy_htod(sld_dev, sld_arr)

    neigh_buf_host = np.zeros(n, dtype=np.int32)
    neigh_buf_dev = cuda.mem_alloc(neigh_buf_host.nbytes)
    cnt_host = np.zeros(1, dtype=np.int32)
    cnt_dev = cuda.mem_alloc(cnt_host.nbytes)

    lib.launch_eps_neighbor(
        ctypes.c_void_p(int(sh_dev)),
        ctypes.c_void_p(int(sld_dev)),
        ctypes.c_int(target_idx),
        ctypes.c_float(eps),
        ctypes.c_float(w),
        ctypes.c_int(SH_DIM),
        ctypes.c_int(n),
        ctypes.c_void_p(int(neigh_buf_dev)),
        ctypes.c_void_p(int(cnt_dev)),
        ctypes.c_int(THREADS_PER_BLOCK)
    )

    cuda.memcpy_dtoh(neigh_buf_host, neigh_buf_dev)
    cuda.memcpy_dtoh(cnt_host, cnt_dev)
    cnt = cnt_host[0]


    sh_dev.free()
    sld_dev.free()
    neigh_buf_dev.free()
    cnt_dev.free()
    return neigh_buf_host[:cnt]