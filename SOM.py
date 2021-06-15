import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

class SOM(object):
    def __init__(self, num_city, data):
        self.num_city = num_city #城市数量
        self.location = data.copy() #城市物理坐标
        self.iteraton = 20000
        self.learning_rate = 0.8 #学习率
        self.dis_mat = self.calculate_dis_mat(num_city, self.location)
        self.best_path = []
        self.best_length = math.inf
        self.iter_x = []
        self.iter_y = []

    def normalize(self, citys):
        #朴素版直接归一化
        x_min, y_min = citys.min(axis=0)[:]
        x_max, y_max = citys.max(axis=0)[:]
        citys[:, 0] = (citys[:, 0] - x_min) / (x_max - x_min)
        citys[:, 1] = (citys[:, 1] - y_min) / (y_max - y_min)
        return citys

    def generate_network(self, size):
        #生成给定大小的神经元网络，返回区间 [0,1] 中二维点的向量。
        return np.random.rand(size, 2)#随机生成符合（0，1）型正态分布的size行2列的向量

    def get_neighborhood(self, center, radix, domain):
        #返回优胜邻域，二维排列就是一个圆网
        """得到一个中心索引周围给定基数的范围高斯。"""
        # 为基数设置上界
        if radix < 1:
            radix = 1
        # 计算圆网到中心的距离
        deltas = np.absolute(center - np.arange(domain))
        distances = np.minimum(deltas, domain - deltas)
        # 计算给定中心周围的高斯分布
        return np.exp(-(distances * distances) / (2 * (radix * radix)))

    def get_route(self, cities, network):#返回神经元的路线,按顺序离城市最近的神经元组成路线
        f = lambda c: self.select_closest(network, c)
        dis = []
        for city in cities:
            dis.append(f(city))
        index = np.argsort(dis)
        #print('index' , index)
        return index #返回路径(城市索引值)

    def select_closest(self, network, city):
        #返回最接近输入点的神经元的索引。
        return np.linalg.norm(network - city, axis=1).argmin()

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

    def som(self):
        citys = self.normalize(self.location) #归一化城市坐标
        n = citys.shape[0] * 8 #输出层神经元的个数：城市个数的八倍
        network = self.generate_network(n) #随机生成【0，1】分布的神经元的位置

        for i in range(self.iteraton):
            index = np.random.randint(self.num_city - 1) #随机选取一个城市作为网络的输入
            city = citys[index] #获取城市归一化后的坐标
            winner_idx = self.select_closest(network, city) #返回最接近输入坐标的神经元的索引。

            #以所选到的神经元为中心，均值为0，方差初始化为城市个数的十分之一创建高斯分布
            gaussian = self.get_neighborhood(winner_idx, n // 10, network.shape[0])
            #所有神经元按照该高斯分布向选中的城市移动
            network += gaussian[:, np.newaxis] * self.learning_rate * (city - network)

            self.learning_rate = self.learning_rate * 0.99997 #学习率的衰减值
            n = n * 0.9997 #高斯分布的方差的衰减值
            if n < 1:  #方差达到阈值结束
                break
            if self.learning_rate < 0.001: #学习率达到阈值结束
                break
            route = self.get_route(citys, network)
            route_l = self.calculate_pathlen(route, self.dis_mat) #计算路径长度
            if route_l < self.best_length: #记录最优路径
                self.best_length = route_l
                self.best_path = route
            self.iter_x.append(i) #记录下收敛过程
            self.iter_y.append(self.best_length)
            print(i, self.iteraton, self.best_length)

            #画出神经网络的规划图
            if not i % 1000:
                plot_network(citys, network, name='SOMgif/{:05d}.png'.format(i))
        plot_network(citys, network, name='SOMgif/final.png')
        return self.best_length, self.best_path

    def run(self):
        self.best_length, self.best_path = self.som()
        return self.best_path, self.location[self.best_path], self.best_length

#绘制神经网络自组织的规划结果图
def plot_network(cities, neurons, name):
    mpl.rcParams['agg.path.chunksize'] = 10000
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    fig = plt.figure(figsize=(5, 5), facecolor='white')
    axis = fig.add_axes([0,0,1,1])

    axis.set_aspect('equal', adjustable='datalim')
    plt.axis('off')

    axis.scatter(cities[:, 0], cities[:, 1], color='red', s=4)
    axis.plot(neurons[:, 0], neurons[:, 1], 'r.', ls='-', color='#0063ba', markersize=2)

    plt.savefig(name , bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

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


data = read_tsp('data/st70.tsp')
data = np.array(data) #结果：[[1.0，64.0，96.0],[],[]...]
data = data[:, 1:] #去除掉第一列

time_start=time.time()  #计算算法运行时间
som = SOM(num_city=data.shape[0], data=data.copy())
Best_pathIndex, Best_path, Best_length = som.run()
print("路径", Best_pathIndex)
time_end=time.time()    #计算算法运行时间
print('time cost',time_end-time_start,'s')

#绘制收敛曲线图
plt.figure()
plt.plot(som.iter_x , som.iter_y)
plt.title('SOM收敛曲线')
plt.xlabel("迭代次数")
plt.ylabel("路径长度")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

#绘制规划结果图
plt.figure()
plt.scatter(Best_path[:, 0], Best_path[:, 1], color='red') #所有点的0坐标做x，1坐标做y ,在plt中标出来
Best_path = np.vstack([Best_path, Best_path[0]]) #将尾和初始连起来
plt.plot(Best_path[:, 0], Best_path[:, 1]) #将所有点按顺序连起来
plt.title('SOM规划结果')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.show()
