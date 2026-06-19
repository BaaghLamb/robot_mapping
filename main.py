import os
import time
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from robot import Robot
from lidar_sensor import Lidar
from occupancy_map import OccupancyGrid
from config import MAP_SIZE, RESOLUTION, LIDAR_RANGE, LIDAR_ANGLE_STEP, LIDAR_NOISE


def compute_accuracy(occ_map, real_map):
    """
    计算地图正确率和未知区域比例
    :param occ_map: OccupancyGrid 实例
    :param real_map: 真实地图 (np.ndarray, 0=空闲, 1=障碍)
    :return: accuracy, unknown_ratio
    """
    built_map = occ_map.get_map(threshold=0.5)
    known_mask = built_map != -1
    if np.sum(known_mask) == 0:
        return 0.0, 1.0
    correct = np.sum((built_map == real_map) & known_mask)
    accuracy = correct / np.sum(known_mask)
    unknown_ratio = 1.0 - np.sum(known_mask) / built_map.size
    return accuracy, unknown_ratio

def save_analysis_table(accuracy, unknown_ratio, build_time, params, save_path):
    with open(save_path, 'a') as f:
        f.write("地图构建数据分析\n")
        f.write(f"实验参数：角度步长 = {params['angle_step']}°，最大测距 = {params['max_range']} m\n")
        f.write(f"栅格分辨率: {params['resolution']} m (固定)\n")
        f.write(f"测距噪声标准差: {params['noise_std']} m (固定)\n")
        f.write(f"扫描次数: {params['scan_count']}\n")
        f.write(f"构建时间: {build_time:.2f} 秒\n")
        f.write(f"地图（已知部分）正确率: {accuracy:.2%}\n")
        f.write(f"未知区域比例: {unknown_ratio:.2%}\n")
    print(f"数据分析表保存至 {save_path}")

def main():
    # 创建结果目录
    os.makedirs('results', exist_ok=True)

    # 初始化环境并保存环境图
    env = Environment()
    fig_env, ax_env = plt.subplots(figsize=(6,6))
    env.plot(ax_env)
    fig_env.savefig('results/environment.png', bbox_inches='tight', dpi=300)
    plt.close(fig_env)
    print("环境图已保存")

    # 生成轨迹
    robot = Robot()
    traj = robot.generate_coverage_trajectory(step=0.5)  
    print(f"预设轨迹点总数: {len(traj)}")

    # 保存轨迹图（在环境图上叠加轨迹）
    fig_traj, ax_traj = plt.subplots(figsize=(6,6))
    env.plot(ax_traj)
    traj_np = np.array(traj)
    ax_traj.plot(traj_np[:, 0], traj_np[:, 1], 'b--', linewidth=1, label='机器人轨迹')
    ax_traj.plot(traj_np[0, 0], traj_np[0, 1], 'go', label='起点')
    ax_traj.plot(traj_np[-1, 0], traj_np[-1, 1], 'ro', label='终点')
    ax_traj.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    fig_traj.tight_layout()
    fig_traj.savefig('results/trajectory.png', bbox_inches='tight', dpi=300)
    plt.close(fig_traj)
    print("轨迹图已保存")

    # 初始化激光雷达、占据栅格地图
    lidar = Lidar(env)  # 使用 config 中的参数
    occ_map = OccupancyGrid()

    # 准备动态可视化
    '''plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title('环境与机器人位姿')
    ax2.set_title('构建中的占据栅格地图')
    fig.tight_layout()'''
    # 准备动态可视化（单图显示，机器人叠加在概率地图上）
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('动态地图构建')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(0, MAP_SIZE[0])
    ax.set_ylim(0, MAP_SIZE[1])

    # 主循环：逐点扫描并更新地图
    start_time = time.time()
    scan_count = 0
    update_interval = 2  # 每2帧更新一次显示

    for i, (x, y, theta) in enumerate(traj):
        # 移动机器人（更新轨迹记录）
        robot.move_to(x, y, theta)
        # 执行激光扫描
        ranges = lidar.scan(x, y, theta)
        # 更新占据栅格地图
        occ_map.update(x, y, theta, ranges, lidar.get_angles_rad(), lidar.max_range)
        scan_count += 1

        if i % update_interval == 0:
            ax.clear()
            # 显示当前概率地图（始终显示最新状态）
            prob_map = occ_map.get_prob_map()
            ax.imshow(prob_map, cmap='gray', origin='lower', vmin=0, vmax=1,
                 extent=[0, MAP_SIZE[0], 0, MAP_SIZE[1]], interpolation='nearest')
    
            # 叠加机器人位姿（红点 + 方向箭头）
            ax.plot(x, y, 'ro', markersize=4, label='机器人')
            ax.arrow(x, y, 0.2*np.cos(theta), 0.2*np.sin(theta),
                head_width=0.15, head_length=0.15, fc='red', ec='red')
    
            # 绘制已走过的轨迹（从 robot.trajectory 获取）
            if len(robot.trajectory) > 1:
                traj_xy = np.array(robot.trajectory)[:, :2]  # 只取 x,y
                ax.plot(traj_xy[:, 0], traj_xy[:, 1], 'b--', linewidth=1.0, alpha=0.6, label='轨迹')
    
            ax.set_title(f'地图构建中 (扫描次数: {scan_count} / 总位姿: {len(traj)})')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_xlim(0, MAP_SIZE[0])
            ax.set_ylim(0, MAP_SIZE[1])
            ax.legend(loc='upper right')
    
            plt.pause(0.01)  # 控制刷新速度

            # 保存过程图（每隔50帧）
            if i % 50 == 0:
                fig.savefig(f'results/process_frame_{i:04d}.png', bbox_inches='tight', dpi=300)

    # 关闭动态模式
    plt.ioff()
    plt.close(fig)

    # 记录构建时间
    build_time = time.time() - start_time
    print(f"地图构建完成，总扫描次数: {scan_count}, 耗时: {build_time:.2f} 秒")

    # 最终地图保存
    final_binary = occ_map.get_map(threshold=0.5)
    # 保存二值图（未知为灰色）
    fig_bin, ax_bin = plt.subplots(figsize=(6,6))
    display = np.ones_like(final_binary, dtype=np.float32) * 0.5
    display[final_binary == 0] = 0.0   # 空闲黑色
    display[final_binary == 1] = 1.0   # 障碍白色
    ax_bin.imshow(display, cmap='gray', origin='lower',
                  extent=[0, MAP_SIZE[0], 0, MAP_SIZE[1]], interpolation='nearest')
    ax_bin.set_title('最终地图 (黑=空闲, 白=障碍, 灰=未知)')
    fig_bin.savefig('results/final_map_binary.png', bbox_inches='tight', dpi=300)
    plt.close(fig_bin)
    print("最终地图已保存")

    # 计算指标
    accuracy, unknown_ratio = compute_accuracy(occ_map, env.real_map)
    print(f"地图正确率: {accuracy:.2%}")
    print(f"未知区域比例: {unknown_ratio:.2%}")

    # 生成数据分析
    params = {
        'resolution': RESOLUTION,
        'angle_step': LIDAR_ANGLE_STEP,
        'max_range': LIDAR_RANGE,         
        'noise_std': LIDAR_NOISE,
        'scan_count': scan_count
    }   
    save_analysis_table(accuracy, unknown_ratio, build_time, params, 'results/analysis_table.txt')

    # 保存传感器扫描效果图（选取一个位姿）
    sample_pose = traj[len(traj)//2]  # 中间位姿
    x_s, y_s, theta_s = sample_pose
    ranges_sample = lidar.scan(x_s, y_s, theta_s)
    fig_scan, ax_scan = plt.subplots(figsize=(8, 8))
    env.plot(ax_scan)
    ax_scan.plot(x_s, y_s, 'ro', markersize=8)
    for r, angle in zip(ranges_sample, lidar.get_angles_rad()):
        world_angle = theta_s + angle
        if r < lidar.max_range - 0.01:
            end_x = x_s + r * np.cos(world_angle)
            end_y = y_s + r * np.sin(world_angle)
            ax_scan.plot([x_s, end_x], [y_s, end_y], 'g-', linewidth=0.5)
            ax_scan.plot(end_x, end_y, 'g.', markersize=2)
        else:
            end_x = x_s + lidar.max_range * np.cos(world_angle)
            end_y = y_s + lidar.max_range * np.sin(world_angle)
            ax_scan.plot([x_s, end_x], [y_s, end_y], 'r--', linewidth=0.3, alpha=0.3)
    ax_scan.set_title(f'激光雷达扫描效果 (位姿: x={x_s:.1f}, y={y_s:.1f}, θ={theta_s:.2f})')
    fig_scan.savefig('results/lidar_scan.png', bbox_inches='tight', dpi=300)
    plt.close(fig_scan)
    print("传感器扫描效果图已保存")


    print("所有结果已保存至 results/ 目录")

if __name__ == "__main__":
    main()