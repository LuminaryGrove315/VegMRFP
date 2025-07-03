import open3d as o3d
import numpy as np
import subprocess
import os

# 配置参数
input_ply = "SpeedPine_Bin.ply"
output_no_color = "output_no_color.mp4"
output_color = "output_color.mp4"
fps = 30
duration = 5
resolution = (1080, 1080)
background_color = [1, 1, 1]  # 黑色背景

# 读取PLY文件
mesh = o3d.io.read_triangle_mesh(input_ply)
has_color = mesh.has_vertex_colors()

# ==================== 关键修改：添加法线信息 ====================
# 方法1：如果原始文件没有法线，则计算顶点法线
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()  # 计算顶点法线

# 方法2：强制重新计算法线（确保方向正确）
mesh.compute_vertex_normals()

# Z轴翻转（保持法线一致性）
mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * [1, -1, 1])
mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals) * [1, -1, 1])
# ==============================================================

# 创建离屏渲染器
vis = o3d.visualization.Visualizer()
vis.create_window(width=resolution[0], height=resolution[1], visible=False)
vis.add_geometry(mesh)

# 设置渲染选项
render_opt = vis.get_render_option()
render_opt.background_color = np.array(background_color)
render_opt.light_on = True  # 开启光照以观察法线效果

# 无颜色版本设置
mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色模型

# 设置相机
ctr = vis.get_view_control()
ctr.set_zoom(0.8)

# 临时帧存储
frame_dir = "temp_frames"
os.makedirs(frame_dir, exist_ok=True)

# 渲染无颜色视频
total_frames = fps * duration
for i in range(total_frames):
    R = mesh.get_rotation_matrix_from_xyz((0, np.pi * 2 / total_frames, 0))
    mesh.rotate(R, center=mesh.get_center())
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    frame_path = os.path.join(frame_dir, f"frame_no_color_{i:04d}.png")
    vis.capture_screen_image(frame_path)

vis.destroy_window()

# 用FFmpeg合成视频
subprocess.run([
    "ffmpeg", "-y", "-framerate", str(fps),
    "-i", os.path.join(frame_dir, "frame_no_color_%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    output_no_color
])

# 如果有顶点颜色，渲染彩色版本
if has_color:
    mesh_color = o3d.io.read_triangle_mesh(input_ply)
    
    # 同样处理法线和翻转
    if not mesh_color.has_vertex_normals():
        mesh_color.compute_vertex_normals()
    mesh_color.compute_vertex_normals()
    mesh_color.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_color.vertices) * [1, -1, 1])
    mesh_color.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh_color.vertex_normals) * [1, -1, 1])
    
    vis_color = o3d.visualization.Visualizer()
    vis_color.create_window(width=resolution[0], height=resolution[1], visible=False)
    vis_color.add_geometry(mesh_color)
    
    render_opt_color = vis_color.get_render_option()
    render_opt_color.light_on = True
    
    for i in range(total_frames):
        R = mesh_color.get_rotation_matrix_from_xyz((0, np.pi * 2 / total_frames, 0))
        mesh_color.rotate(R, center=mesh_color.get_center())
        vis_color.update_geometry(mesh_color)
        vis_color.poll_events()
        vis_color.update_renderer()
        frame_path = os.path.join(frame_dir, f"frame_color_{i:04d}.png")
        vis_color.capture_screen_image(frame_path)
    
    vis_color.destroy_window()
    
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_color_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        output_color
    ])

# 清理临时文件
for file in os.listdir(frame_dir):
    os.remove(os.path.join(frame_dir, file))
os.rmdir(frame_dir)

print(f"视频生成完成：{output_no_color}" + (f", {output_color}" if has_color else ""))