# 所有参数配置（地图、雷达、机器人）

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]   # 黑体

# 地图参数
MAP_SIZE = (10, 10)          # 地图大小
RESOLUTION = 0.05             # 栅格分辨率 
MAP_WIDTH = int(MAP_SIZE[0] / RESOLUTION)
MAP_HEIGHT = int(MAP_SIZE[1] / RESOLUTION)

# 雷达参数 
LIDAR_RANGE = 5.0            # 最大测距 #对比 0.5 2.0 5.0 
LIDAR_ANGLE_MIN = -180       # 雷达起始角度
LIDAR_ANGLE_MAX = 180        # 雷达结束角度
LIDAR_ANGLE_STEP = 5     # 角度步长  #雷达角分辨率对比  2 5 10
LIDAR_NOISE = 0.05        # 测距噪声标准差  

# 机器人参数 
ROBOT_START = (2.0, 1.0, 0)  # 初始位姿(x,y,theta)，单位米和弧度