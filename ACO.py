import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
    
class ACO(object):
    def __init__(self, num_city, data):
        self.m = math.ceil(num_city * 2) # 蚂蚁数量，一般设置为目标数的1.5倍
        self.alpha = 1 # 信息素重要程度因子，一般[1 , 4]
        self.beta = 5#4  # 启发函数重要因子，一般[3 , 4.5]
        self.rho = 0.1 #0.2  # 信息素挥发因子
        self.Q = 1#20  # 信息素常量
        self.num_city = num_city  # 城市规模
        self.location = data  # 城市坐标
        self.Tau = np.ones([num_city, num_city])  # 信息素矩阵,Tau(i, j)表示路径(i, j)的信息素量,设置为1防止/0错误
        self.Table = [[0 for _ in range(num_city)] for _ in range(self.m)]  # 生成的蚁群,路径记录表,第一列是所有蚂蚁的起点城市
        self.iter_max = 200 #200#迭代最大次数
        self.dis_mat = self.calculate_dis_mat(num_city, self.location)  # 计算城市之间的距离矩阵
        self.Eta = 10. / self.dis_mat  # 启发式函数
        #self.Eta = 1. / self.dis_mat  # 启发式函数
        self.paths = None  # 蚁群中每个个体的长度
        # 存储存储每次迭代的最佳路径，画出收敛图
        self.iter_x = []
        self.iter_y = []
        # 存储每次迭代的最佳路径选择，画出规划结果
        self.iter_path = []
    # 轮盘赌选择
    def rand_choose(self, p):
        x = np.random.rand()
        for i, t in enumerate(p):
            x -= t
            if x <= 0:
                break
        return i

    # 生成蚁群
    def get_ants(self, num_city):
        for i in range(self.m):
            start = np.random.randint(num_city - 1)
            #start = np.random.randint(1,num_city) #设计直接从0出发，最终回到0。
            self.Table[i][0] = start    #将各个蚂蚁随机地放置在不同的出发点。
            unvisit = list([x for x in range(num_city) if x != start]) #把还没走到的点加入
            #unvisit = list([x for x in range(1,num_city) if x != start])
            current = start
            j = 1
            while len(unvisit) != 0: #蚂蚁访问完所有的城市
                P = []
                # 通过信息素计算城市之间的转移概率
                for v in unvisit:
                    P.append(self.Tau[current][v] ** self.alpha * self.Eta[current][v] ** self.beta)
                P_sum = sum(P)
                P = [x / P_sum for x in P] #归一化
                # 轮盘赌选择一个一个城市
                index = self.rand_choose(P)
                current = unvisit[index]
                self.Table[i][j] = current
                unvisit.remove(current)
                j += 1

    # 计算不同城市之间的距离
    def calculate_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算一条路径的长度
    def calculate_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算一个群体的长度
    def calculate_paths(self, paths):
        result = []
        for one in paths:
            length = self.calculate_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 更新信息素
    def update_Tau(self):
        delta_tau = np.zeros([self.num_city, self.num_city])
        for i in range(self.m):
            for j in range(self.num_city - 1):
                a = self.Table[i][j]
                b = self.Table[i][j + 1]
                delta_tau[a][b] = delta_tau[a][b] + self.Q / self.paths[i]
            a = self.Table[i][0]
            b = self.Table[i][-1]
            delta_tau[a][b] = delta_tau[a][b] + self.Q / self.paths[i]
        self.Tau = (1 - self.rho) * self.Tau + delta_tau

    def aco(self):
        best_lenth = math.inf
        best_path = None
        for cnt in range(self.iter_max):
            # 生成新的蚁群
            self.get_ants(self.num_city)  # 输出>>self.Table，蚂蚁访问完所有的城市
            #print(self.Table)
            self.paths = self.calculate_paths(self.Table)
            # 取该蚁群的最优解
            tmp_lenth = min(self.paths)
            tmp_path = self.Table[self.paths.index(tmp_lenth)]
            # 可视化每一次迭代的最优路径，从而绘制出路径变化的整个过程
            # 更新最优解
            if tmp_lenth < best_lenth:
                best_lenth = tmp_lenth
                best_path = tmp_path
                #记录下每次迭代选到的最优路径规划图
                self.iter_path.append(best_path)
            # 更新信息素
            self.update_Tau()

            # 保存结果
            self.iter_x.append(cnt)
            self.iter_y.append(best_lenth)
            print(cnt,best_lenth)
        return best_lenth, best_path

    def run(self):
        best_length, best_path = self.aco()
        print('路径：' , best_path)
        return best_path, self.location[best_path], best_length


# 读取数据
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data

data = read_tsp('data/att48.tsp') #结果：[[1.0，64.0，96.0],[],[]...]
data = np.array(data)#将数据转换为矩阵
data = data[:, 1:]#去除掉第一列

time_start=time.time()  #计算算法运行时间
aco = ACO(num_city=data.shape[0], data=data.copy())
Best_pathIndex, Best_path, Best = aco.run()
print("路径", Best_path)
time_end=time.time()    #计算算法运行时间
print('time cost',time_end-time_start,'s')

#绘制收敛曲线图
plt.figure()
plt.plot(aco.iter_x , aco.iter_y)
plt.title('蚁群算法收敛曲线')
plt.xlabel("迭代次数")
plt.ylabel("路径长度")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

#绘制规划结果图
plt.figure()
# 打开交互模式
plt.ion()
for i in range(len(aco.iter_path)):
    # 清除原有图像
    plt.cla()
    plt.scatter(Best_path[:, 0], Best_path[:, 1],color='red')  # 所有点的0坐标做x，1坐标做y ,在plt中标出来
    # 使用annotate函数必备参数绘制注解
    # for index, point in zip(Best_pathIndex, Best_path):
    #     plt.annotate(index + 1, xy=(point[0], point[1]))
    plt.title('蚁群算法规划结果')
    plt.xlabel("点横坐标")
    plt.ylabel("点纵坐标")

    iter_path = aco.location[aco.iter_path[i]]
    #print(iter_path)
    iter_path = np.vstack([iter_path, iter_path[0]])  # 将尾和初始连起来
    plt.plot(iter_path[:, 0], iter_path[:, 1])  # 将所有点按顺序连起来
    plt.savefig(r'D:/.c学习资料/人工智能/TSP_collection-master/GA_ACOgif\{}.png'.format(i))
    # 暂停
    plt.pause(0.1)
# 关闭交互模式
plt.ioff()
plt.show()
