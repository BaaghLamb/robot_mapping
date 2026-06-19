# 机器人运动与位姿管理

import numpy as np
from config import ROBOT_START, MAP_SIZE
from environment import Environment

class Robot:
    def __init__(self):
        self.x, self.y, self.theta = ROBOT_START
        self.trajectory = [(self.x, self.y, self.theta)]

    def move_to(self, x, y, theta):
        self.x = np.clip(x, 0, MAP_SIZE[0])
        self.y = np.clip(y, 0, MAP_SIZE[1])
        self.theta = theta % (2 * np.pi)
        self.trajectory.append((self.x, self.y, self.theta))

    def generate_coverage_trajectory(self, step=0.5):
        traj = []

        y = 1.0
        for x in np.arange(2.0, 9.5, step):
            traj.append((x, y, 0))

        x = 9.5
        for y in np.arange(1.0, 2.0, step):
            traj.append((x, y, np.pi/2)) 

        y = 2.0
        for x in np.arange(9.5, 4.0, -step):
            traj.append((x, y, np.pi))

        x = 4.0
        for y in np.arange(2.0, 3.0, step):
            traj.append((x, y, np.pi/2))

        y = 3.0
        for x in np.arange(4.0, 9.5, step):
            traj.append((x, y, 0))

        x = 9.5
        for y in np.arange(3.0, 4.0, step):
            traj.append((x, y, np.pi/2))   

        y = 4.0
        for x in np.arange(9.5, 2.5, -step):
            traj.append((x, y, np.pi))

        x = 2.5
        for y in np.arange(4.0, 6.0, step):
            traj.append((x, y, np.pi/2))

        y = 6.0
        for x in np.arange(2.5, 4.5, step):
            traj.append((x, y, 0))

        x = 4.5
        for y in np.arange(6.0, 7.0, step):
            traj.append((x, y, np.pi/2))  

        y = 7.0
        for x in np.arange(4.5, 0.5, -step):
            traj.append((x, y, np.pi))

        x = 0.5
        for y in np.arange(7.0, 8.0, step):
            traj.append((x, y, np.pi/2))

        y = 8.0
        for x in np.arange(0.5, 4.5, step):
            traj.append((x, y, 0))

        x = 4.5
        for y in np.arange(8.0, 9.0, step):
            traj.append((x, y, np.pi/2))
        
        y = 9.0
        for x in np.arange(4.5, 0.5, -step):
            traj.append((x, y, np.pi))

        x = 0.5
        for y in np.arange(9.0, 9.5, step):
            traj.append((x, y, np.pi/2))

        y = 9.5
        for x in np.arange(0.5, 8.0, step):
            traj.append((x, y, 0))

        x = 8.0
        for y in np.arange(9.5, 5.0, -step):
            traj.append((x, y, 3*np.pi/2)) 

        y = 5.0
        for x in np.arange(8.0, 9.5, step):
            traj.append((x, y, 0))

        x = 9.5
        for y in np.arange(5.0, 6.0, step):
            traj.append((x, y, np.pi/2)) 

        y = 6.0
        for x in np.arange(9.5, 6.5, -step):
            traj.append((x, y, np.pi))

        x = 6.5
        for y in np.arange(6.0, 9.0, step):
            traj.append((x, y, np.pi/2)) 

        return traj

    def plot_trajectory(self, ax):
        traj_np = np.array(self.trajectory)
        ax.plot(traj_np[:, 0], traj_np[:, 1], "b--", linewidth=1, label="机器人运动轨迹")
        ax.scatter(self.x, self.y, c="blue", marker="o", s=40, label="机器人当前位置")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        return ax

if __name__ == "__main__":
    robot = Robot()
    env = Environment()
    traj = robot.generate_coverage_trajectory()
    print(f"轨迹点总数: {len(traj)}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    env.plot(ax)
    for pos in traj:
        robot.move_to(*pos)
    robot.plot_trajectory(ax)
    plt.title("机器人运动轨迹")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig("results/trajectory.png", bbox_inches="tight")
    plt.show()