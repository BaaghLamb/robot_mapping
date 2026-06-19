# 二维仿真环境定义（墙体+障碍物）

import numpy as np
import matplotlib.pyplot as plt
from config import MAP_SIZE, RESOLUTION, MAP_WIDTH, MAP_HEIGHT

class Environment:
    def __init__(self):
        self.real_map = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.int8)  # 0=空闲, 1=障碍
        self._build_map()

    def _build_map(self):
        """构建墙体+通道+3个障碍物"""
        # 边界墙体
        self.real_map[0, :] = 1
        self.real_map[-1, :] = 1
        self.real_map[:, 0] = 1
        self.real_map[:, -1] = 1

        # 自定义障碍物（3个）
        #障碍物1
        self.add_rect_obstacle(x1=0.5, y1=0.5, x2=1.5, y2=5.5)
        self.add_rect_obstacle(x1=0.5, y1=1.5, x2=3.5, y2=2.5)

        #障碍物2
        self.add_rect_obstacle(x1=5, y1=5.5, x2=6, y2=9.0)
        self.add_rect_obstacle(x1=3.5, y1=4.5, x2=7.5, y2=5.5)
        
        #障碍物3
        self.add_rect_obstacle(x1=8.5, y1=6.5, x2=9.5, y2=9.5)
        
    def add_rect_obstacle(self, x1, y1, x2, y2):
        """
        快速添加矩形障碍物（世界坐标）
        x1,y1: 矩形左下角坐标(m)
        x2,y2: 矩形右上角坐标(m)
        """
        gx1, gy1 = self.world_to_grid(x1, y1)
        gx2, gy2 = self.world_to_grid(x2, y2)
        # 防止越界
        gx1, gx2 = max(0, gx1), min(MAP_WIDTH-1, gx2)
        gy1, gy2 = max(0, gy1), min(MAP_HEIGHT-1, gy2)
        self.real_map[gy1:gy2+1, gx1:gx2+1] = 1

    def world_to_grid(self, x, y):
        """世界坐标转栅格坐标"""
        gx = int(x / RESOLUTION)
        gy = int(y / RESOLUTION)
        return gx, gy

    def grid_to_world(self, gx, gy):
        """栅格坐标转世界坐标"""
        x = gx * RESOLUTION
        y = gy * RESOLUTION
        return x, y

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(self.real_map, cmap="gray", origin="lower",
                  extent=[0, MAP_SIZE[0], 0, MAP_SIZE[1]])
        ax.set_title("仿真环境图")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        return ax

if __name__ == "__main__":
    env = Environment()
    ax = env.plot()
    plt.savefig("results/environment.png")
    plt.show()