// cuda_kernels.cu
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdio.h>

typedef float f32;
typedef int i32;

__device__ __forceinline__ f32 euclid3d(f32 x1, f32 y1, f32 z1, f32 x2, f32 y2, f32 z2)
{
    f32 dx = x1 - x2;
    f32 dy = y1 - y2;
    f32 dz = z1 - z3;
    return sqrt(dx*dx + dy*dy + dz*dz);
}


__global__ void compute_sld_kernel(
    const f32* __restrict xyz,
    f32* __restrict sld_out,
    i32 n_points,
    i32 k_nn
)
{
    i32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_points) return;

    f32 xi = xyz[tid * 3 + 0];
    f32 yi = xyz[tid * 3 + 1];
    f32 zi = xyz[tid * 3 + 2];

    f32 min_dists[16];
    for (i32 t = 0; t < k_nn; t++) min_dists[t] = 1e9f;

    for (i32 j = 0; j < n_points; j++)
    {
        if (tid == j) continue;
        f32 xj = xyz[j * 3 + 0];
        f32 yj = xyz[j * 3 + 1];
        f32 zj = xyz[j * 3 + 2];
        f32 d = euclid3d(xi, yi, zi, xj, yj, zj);

        i32 pos = k_nn;
        for (i32 t = 0; t < k_nn; t++)
        {
            if (d < min_dists[t]){ pos = t; break; }
        }
        if (pos < k_nn)
        {
            for (i32 s = k_nn - 1; s > pos; s--)
                min_dists[s] = min_dists[s - 1];
            min_dists[pos] = d;
        }
    }

    f32 dist_sum = 0.0f;
    for (i32 t = 0; t < k_nn; t++) dist_sum += min_dists[t];
    sld_out[tid] = (f32)k_nn / dist_sum;
}


__device__ f32 weighted_distance_dev(
    const f32* sh_i, f32 sld_i,
    const f32* sh_j, f32 sld_j,
    f32 w, i32 sh_dim
)
{
    f32 sum_sh_i = 0.f, sum_sh_j = 0.f;
    for (i32 d = 0; d < sh_dim; d++)
    {
        sum_sh_i += sh_i[d];
        sum_sh_j += sh_j[d];
    }
    f32 feat_i = (w / sh_dim) * sum_sh_i + (1.f - w) * sld_i;
    f32 feat_j = (w / sh_dim) * sum_sh_j + (1.f - w) * sld_j;
    return fabs(feat_i - feat_j);
}


__global__ void get_eps_neighbor_kernel(
    const f32* sh_mat, const f32* sld_arr,
    i32 target_idx, f32 eps, f32 w, i32 sh_dim,
    i32 n_points,
    i32* neighbor_buf, i32* out_cnt
)
{
    i32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) return;

    const f32* sh_p = sh_mat + target_idx * sh_dim;
    f32 sld_p = sld_arr[target_idx];
    i32 cnt = 0;

    for (i32 q = 0; q < n_points; q++)
    {
        if (q == target_idx) continue;
        const f32* sh_q = sh_mat + q * sh_dim;
        f32 sld_q = sld_arr[q];
        f32 dist = weighted_distance_dev(sh_p, sld_p, sh_q, sld_q, w, sh_dim);
        if (dist <= eps)
        {
            neighbor_buf[cnt++] = q;
        }
    }
    *out_cnt = cnt;
}


extern "C"
{
    void launch_sld_kernel(f32* xyz_dev, f32* sld_dev, i32 n, i32 k, i32 block_size)
    {
        dim3 block(block_size);
        dim3 grid((n + block.x - 1) / block.x);
        compute_sld_kernel<<<grid, block>>>(xyz_dev, sld_dev, n, k);
        cudaDeviceSynchronize();
    }

    void launch_eps_neighbor(
        f32* sh_dev, f32* sld_dev,
        i32 target, f32 eps, f32 w, i32 sh_dim, i32 n,
        i32* neigh_buf_dev, i32* cnt_dev, i32 block_size
    )
    {
        dim3 block(block_size);
        dim3 grid(1);
        get_eps_neighbor_kernel<<<grid, block>>>(
            sh_dev, sld_dev, target, eps, w, sh_dim, n, neigh_buf_dev, cnt_dev
        );
        cudaDeviceSynchronize();
    }
}