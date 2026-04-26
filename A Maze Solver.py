import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import heapq


class Node:
    """表示搜索树中的一个节点"""

    def __init__(self, position, parent=None):
        self.position = position  # (x, y) 坐标
        self.parent = parent  # 父节点
        self.g = 0  # 从起点到当前节点的实际代价
        self.h = 0  # 从当前节点到终点的估计代价
        self.f = 0  # 总评估代价 f = g + h

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        return f"Node({self.position}, g={self.g}, h={self.h}, f={self.f})"


class AStarMazeSolver:
    """使用A*算法求解迷宫寻路问题"""

    def __init__(self, maze, start, end):
        """
        初始化求解器

        参数:
            maze: 二维数组表示的迷宫，0表示可通过，1表示障碍
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
        """
        self.maze = np.array(maze)
        self.start = start
        self.end = end
        self.rows, self.cols = self.maze.shape

        # 定义移动方向 (上, 右, 下, 左)
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    def heuristic(self, position):
        """启发式函数：曼哈顿距离"""
        return abs(position[0] - self.end[0]) + abs(position[1] - self.end[1])

    def is_valid_position(self, position):
        """检查位置是否有效（在迷宫范围内且不是障碍物）"""
        x, y = position
        return 0 <= x < self.cols and 0 <= y < self.rows and self.maze[y, x] == 0

    def get_neighbors(self, node):
        """获取当前节点的所有合法邻居节点"""
        neighbors = []
        x, y = node.position

        for dx, dy in self.directions:
            new_x, new_y = x + dx, y + dy
            new_position = (new_x, new_y)

            if self.is_valid_position(new_position):
                neighbors.append(Node(new_position, node))

        return neighbors

    def solve(self):
        """执行A*算法求解迷宫路径"""
        # 初始化起点和终点节点
        start_node = Node(self.start)
        end_node = Node(self.end)

        # 创建开放列表和关闭列表
        open_list = []
        closed_list = []

        # 将起点加入开放列表
        heapq.heappush(open_list, start_node)

        # 记录每个位置的最佳g值
        g_scores = {self.start: 0}

        # 记录节点与其位置的关系
        node_positions = {self.start: start_node}

        while open_list:
            # 获取f值最小的节点
            current_node = heapq.heappop(open_list)

            # 如果到达终点，回溯路径
            if current_node == end_node:
                return self.reconstruct_path(current_node)

            # 将当前节点加入关闭列表
            closed_list.append(current_node)

            # 生成所有邻居节点
            neighbors = self.get_neighbors(current_node)

            for neighbor in neighbors:
                # 如果邻居节点已在关闭列表中，跳过
                if neighbor in closed_list:
                    continue

                # 计算从起点到邻居节点的实际代价
                neighbor.g = current_node.g + 1
                neighbor.h = self.heuristic(neighbor.position)
                neighbor.f = neighbor.g + neighbor.h

                # 如果邻居节点不在开放列表中，或者找到了更优路径
                if neighbor.position not in g_scores or neighbor.g < g_scores[neighbor.position]:
                    g_scores[neighbor.position] = neighbor.g
                    node_positions[neighbor.position] = neighbor

                    # 如果邻居节点不在开放列表中，加入开放列表
                    if neighbor not in open_list:
                        heapq.heappush(open_list, neighbor)

        # 如果开放列表为空仍未找到路径，返回None
        return None

    def reconstruct_path(self, node):
        """从终点节点回溯重建路径"""
        path = []
        current = node

        while current is not None:
            path.append(current.position)
            current = current.parent

        path.reverse()
        return path

    def visualize(self, path=None):
        """可视化迷宫和路径"""
        fig, ax = plt.subplots(figsize=(8, 8))

        # 创建自定义颜色映射
        cmap = ListedColormap(['white', 'black', 'green', 'red', 'blue'])

        # 创建可视化矩阵
        visualization = self.maze.copy().astype(float)

        # 标记起点和终点
        start_y, start_x = self.start[1], self.start[0]
        end_y, end_x = self.end[1], self.end[0]
        visualization[start_y, start_x] = 2  # 起点用绿色表示
        visualization[end_y, end_x] = 3  # 终点用红色表示

        # 如果有路径，标记路径
        if path:
            for x, y in path[1:-1]:  # 排除起点和终点
                visualization[y, x] = 4  # 路径用蓝色表示

        # 绘制迷宫
        ax.imshow(visualization, cmap=cmap)

        # 添加网格线
        ax.grid(which='major', axis='both', linestyle='-', color='gray', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, self.cols, 1))
        ax.set_yticks(np.arange(-0.5, self.rows, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # 添加坐标标签
        for i in range(self.rows):
            for j in range(self.cols):
                ax.text(j, i, f'({j},{i})', ha='center', va='center', fontsize=8, color='red')

        # 设置标题
        if path:
            ax.set_title(f'A* Algorithm: Path Found (Length: {len(path)})')
        else:
            ax.set_title('A* Algorithm: No Path Found')

        plt.tight_layout()
        plt.show()


def main():
    # 定义迷宫 (0表示可通过，1表示障碍物)
    maze = [
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ]

    # 定义起点和终点 (注意：坐标格式为(x, y)，从0开始)
    start = (0, 0)  # 对应题目中的(1,1)，因为题目是从1开始计数
    end = (4, 4)  # 对应题目中的(5,5)

    # 创建求解器
    solver = AStarMazeSolver(maze, start, end)

    # 求解路径
    path = solver.solve()

    # 输出结果
    print("迷宫地图:")
    for row in maze:
        print(row)

    print(f"\n起点: {start}")
    print(f"终点: {end}")

    if path:
        print(f"\n找到路径! 路径长度: {len(path)}")
        print("路径坐标:")
        for i, pos in enumerate(path):
            print(f"步骤 {i}: {pos}")
    else:
        print("\n未找到可行路径!")

    # 可视化结果
    solver.visualize(path)


if __name__ == "__main__":
    main()