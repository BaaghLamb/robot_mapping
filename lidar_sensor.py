# lidar_sensor.py
# 模拟二维激光雷达，支持角度范围、分辨率、最大距离和测距噪声

import numpy as np
from config import (
    LIDAR_RANGE,
    LIDAR_ANGLE_MIN,
    LIDAR_ANGLE_MAX,
    LIDAR_ANGLE_STEP,
    LIDAR_NOISE,
    RESOLUTION,
    MAP_WIDTH,
    MAP_HEIGHT
)

class Lidar:
    def __init__(self, env, angle_min=None, angle_max=None, angle_step=None, max_range=None, noise_std=None):
        """
        激光雷达模拟器
        :param env: Environment 对象，包含真实地图
        :param angle_min: 起始角度（度），默认从 config 读取
        :param angle_max: 结束角度（度），默认从 config 读取
        :param angle_step: 角度步长（度），默认从 config 读取
        :param max_range: 最大探测距离（米），默认从 config 读取
        :param noise_std: 测距高斯噪声标准差（米），默认从 config 读取
        """
        self.env = env
        # 参数初始化，若未传入则使用config中的值
        self.angle_min_deg = angle_min if angle_min is not None else LIDAR_ANGLE_MIN
        self.angle_max_deg = angle_max if angle_max is not None else LIDAR_ANGLE_MAX
        self.angle_step_deg = angle_step if angle_step is not None else LIDAR_ANGLE_STEP
        self.max_range = max_range if max_range is not None else LIDAR_RANGE
        self.noise_std = noise_std if noise_std is not None else LIDAR_NOISE

        # 角度转换为弧度，并生成所有射线角度（相对于机器人朝向）
        self.angles_deg = np.arange(self.angle_min_deg, self.angle_max_deg + self.angle_step_deg, self.angle_step_deg)
        self.angles_rad = np.deg2rad(self.angles_deg)
        self.num_beams = len(self.angles_rad)

        # 射线投射步长（米），可根据地图分辨率调整
        self.step_size = RESOLUTION * 0.5  # 使用分辨率的一半，保证不漏检

    def scan(self, x, y, theta):
        """
        在机器人位姿 (x, y, theta) 处进行一次扫描
        :param x, y: 世界坐标（米）
        :param theta: 机器人朝向（弧度）
        :return: 距离列表（米），长度等于光束数，超出最大距离则返回 max_range
        """
        ranges = []
        for angle in self.angles_rad:
            # 世界坐标系下的射线角度
            world_angle = theta + angle
            # 沿射线步进检测
            dist = 0.0
            hit = False
            while dist < self.max_range:
                dist += self.step_size
                # 计算当前端点世界坐标
                px = x + dist * np.cos(world_angle)
                py = y + dist * np.sin(world_angle)
                # 转换为栅格坐标
                gx, gy = self.env.world_to_grid(px, py)
                # 检查是否越界
                if gx < 0 or gx >= MAP_WIDTH or gy < 0 or gy >= MAP_HEIGHT:
                    hit = True   # 遇到边界视为碰撞，距离为当前 dist
                    break
                # 检查是否为障碍物
                if self.env.real_map[gy, gx] == 1:
                    hit = True
                    break
            if hit:
                # 检测到障碍物，加入高斯噪声（如果有）
                if self.noise_std > 0:
                    dist += np.random.normal(0, self.noise_std)
                    # 保证距离非负且不超过最大范围
                    dist = max(0.0, min(dist, self.max_range))
                ranges.append(dist)
            else:
                # 未检测到障碍物，返回最大距离（代表无穷远）
                ranges.append(self.max_range)
        return ranges

    def get_angles_rad(self):
        """返回所有射线的相对角度（弧度）"""
        return self.angles_rad.copy()

    def get_angles_deg(self):
        """返回所有射线的相对角度（度）"""
        return self.angles_deg.copy()


# 简单的测试代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment import Environment

    # 创建环境
    env = Environment()
    # 创建激光雷达（使用config默认参数）
    lidar = Lidar(env)

    # 设定机器人位姿（在自由空间内）
    robot_x, robot_y, robot_theta = 2.0, 2.0, 0.0

    # 执行一次扫描
    ranges = lidar.scan(robot_x, robot_y, robot_theta)

    # 可视化扫描结果
    fig, ax = plt.subplots(figsize=(8, 8))
    env.plot(ax)
    # 绘制机器人位置
    ax.plot(robot_x, robot_y, 'ro', markersize=8)
    # 绘制每条射线
    for r, angle in zip(ranges, lidar.get_angles_rad()):
        world_angle = robot_theta + angle
        if r < lidar.max_range - 0.01:  # 只绘制有效探测点
            end_x = robot_x + r * np.cos(world_angle)
            end_y = robot_y + r * np.sin(world_angle)
            ax.plot([robot_x, end_x], [robot_y, end_y], 'g-', linewidth=0.5)
            ax.plot(end_x, end_y, 'g.', markersize=2)
        else:
            # 未探测到，绘制一条到最大距离的虚线（可选）
            end_x = robot_x + lidar.max_range * np.cos(world_angle)
            end_y = robot_y + lidar.max_range * np.sin(world_angle)
            ax.plot([robot_x, end_x], [robot_y, end_y], 'r--', linewidth=0.3, alpha=0.3)

    ax.set_title(f"激光雷达扫描效果 (位姿: x={robot_x}, y={robot_y}, θ={robot_theta:.2f} rad)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig("results/lidar_scan.png", bbox_inches="tight")
    plt.show()