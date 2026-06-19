# 占据栅格地图构建（基于二值贝叶斯滤波 / Log-Odds）

import numpy as np
from config import MAP_WIDTH, MAP_HEIGHT, RESOLUTION

class OccupancyGrid:
    def __init__(self, width=MAP_WIDTH, height=MAP_HEIGHT, resolution=RESOLUTION,
                 prior=0.5, log_odds_occ=0.9, log_odds_free=0.7):
        """
        初始化占据栅格地图
        :param width, height: 栅格数量
        :param resolution: 地图分辨率（米/格）
        :param prior: 初始占据概率 (0~1)
        :param log_odds_occ: 观测到障碍时的更新增量（正数）
        :param log_odds_free: 观测到空闲时的更新增量（正数，实际减去）
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.prior = prior
        self.log_odds_occ = log_odds_occ
        self.log_odds_free = log_odds_free

        # 对数几率矩阵，初始为 log(prior/(1-prior))
        init_log_odds = np.log(prior / (1 - prior + 1e-9))
        self.log_odds = np.full((height, width), init_log_odds, dtype=np.float32)

        # 计数器
        self.occ_counts = np.zeros((height, width), dtype=np.int32)
        self.free_counts = np.zeros((height, width), dtype=np.int32)

        # 快速获取概率图（二值/概率）
        self.prob_map = None


    def update(self, robot_x, robot_y, robot_theta, ranges, angles, max_range):
        """
        根据一次激光扫描更新地图（带机器人朝向）
        :param robot_x, robot_y: 机器人世界坐标 (米)
        :param robot_theta: 机器人朝向 (弧度)
        :param ranges: 各光束测距值 (米)
        :param angles: 各光束相对机器人朝向的角度 (弧度)
        :param max_range: 激光最大测距 (米)
        """
        # 遍历每束激光
        for r, angle in zip(ranges, angles):
            # 世界坐标系下的射线角度
            world_angle = robot_theta + angle

            # 若检测到障碍（r < max_range），更新端点障碍信息
            if r < max_range - 0.01:
                # 端点世界坐标
                end_x = robot_x + r * np.cos(world_angle)
                end_y = robot_y + r * np.sin(world_angle)
                gx, gy = self._world_to_grid(end_x, end_y)
                if self._is_valid_grid(gx, gy):
                    self._update_cell(gx, gy, is_occupied=True)
                    self.occ_counts[gy, gx] += 1

                # 射线路径上的栅格（从机器人到端点前）标记为空闲
                # 使用步进方式，步长略小于分辨率
                step = self.resolution * 0.5
                dist = 0.0
                while dist < r - step:
                    dist += step
                    px = robot_x + dist * np.cos(world_angle)
                    py = robot_y + dist * np.sin(world_angle)
                    gx_path, gy_path = self._world_to_grid(px, py)
                    if self._is_valid_grid(gx_path, gy_path):
                        # 避免将端点栅格重复标记
                        if not (gx_path == gx and gy_path == gy):
                            self._update_cell(gx_path, gy_path, is_occupied=False)
                            self.free_counts[gy_path, gx_path] += 1
            else:
                # 未检测到障碍（超出最大距离），将整条射线路径标记为空闲（直到最大距离）
                step = self.resolution * 0.5
                dist = 0.0
                while dist < max_range - step:
                    dist += step
                    px = robot_x + dist * np.cos(world_angle)
                    py = robot_y + dist * np.sin(world_angle)
                    gx, gy = self._world_to_grid(px, py)
                    if self._is_valid_grid(gx, gy):
                        self._update_cell(gx, gy, is_occupied=False)
                        self.free_counts[gy, gx] += 1

    def _update_cell(self, gx, gy, is_occupied):
        """更新单个栅格的对数几率"""
        if is_occupied:
            self.log_odds[gy, gx] += self.log_odds_occ
        else:
            self.log_odds[gy, gx] -= self.log_odds_free
        # 限制范围防止数值溢出
        self.log_odds[gy, gx] = np.clip(self.log_odds[gy, gx], -10.0, 10.0)

    def get_probability(self, gx, gy):
        """获取栅格占据概率 (0~1)"""
        log_odds = self.log_odds[gy, gx]
        prob = 1.0 - 1.0 / (1.0 + np.exp(log_odds))
        return prob
    
    def get_prob_map(self):
        """返回概率图矩阵 (0~1)，所有栅格均有概率值（包括未知区域）"""
        prob_map = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
        return prob_map

    def get_map(self, threshold=0.5, return_prob=False):
        """
        返回地图矩阵
        :param threshold: 二值化阈值，概率大于此值为障碍
        :param return_prob: 若True返回概率图，否则返回二值图（-1未知, 0空闲, 1障碍）
        """
        if return_prob:
            return self.get_prob_map()
        else:
            prob_map = self.get_prob_map()
            # 二值化：根据访问标记判断未知区域
            visited = (self.occ_counts + self.free_counts) > 0
            binary_map = np.full_like(prob_map, -1, dtype=np.int8)  # -1表示未知
            binary_map[visited] = (prob_map[visited] >= threshold).astype(np.int8)
            return binary_map
        

    def _world_to_grid(self, x, y):
        """世界坐标转栅格坐标"""
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        return gx, gy

    def _is_valid_grid(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    def _robot_angle_to_world(self, relative_angle, robot_theta):
        """已不再使用，统一在update内部计算"""
        pass

    def plot(self, ax=None, threshold=0.5, show_unknown=True):
        """可视化当前地图"""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        # 获取二值地图（带未知）
        binary_map = self.get_map(threshold=threshold)

        display_map = np.ones_like(binary_map, dtype=np.float32) * 0.5  
        display_map[binary_map == 0] = 0.0   
        display_map[binary_map == 1] = 1.0   
        ax.imshow(display_map, cmap='gray', origin='lower',
                  extent=[0, self.width*self.resolution, 0, self.height*self.resolution])
        ax.set_title("占据栅格地图")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        return ax


# 测试代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from environment import Environment
    from lidar_sensor import Lidar

    # 创建环境和雷达
    env = Environment()
    lidar = Lidar(env)

    # 初始化占据栅格地图
    occ_map = OccupancyGrid()

    # 模拟多次扫描（不同位姿）
    poses = [(2.0, 2.0, 0.0), (4.0, 3.0, 0.5), (6.0, 7.0, -0.3), (8.0, 8.0, 1.2)]
    for x, y, theta in poses:
        ranges = lidar.scan(x, y, theta)
        # 更新地图（传入机器人朝向）
        occ_map.update(x, y, theta, ranges, lidar.get_angles_rad(), lidar.max_range)

    # 可视化最终地图
    fig, ax = plt.subplots(figsize=(6,6))
    occ_map.plot(ax)
    plt.tight_layout()
    plt.savefig("results/occupancy_map_test.png", bbox_inches="tight")
    plt.show()