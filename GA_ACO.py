import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# 存储存储每次迭代的最佳路径，画出收敛图
iter_x = []
iter_y = []
# 存储每次迭代的最佳路径选择，画出规划结果
iter_path = []

class GA(object):
    def __init__(self, num_city, num_total, iteration, data, aco_init): #
        self.num_city = num_city  # 城市数量
        self.num_total = num_total  # 种群中个体数（等于蚁群数量)
        self.fitnesses = []  # 适应值
        self.iteration = iteration  # 迭代轮数
        self.location = data  # 城市坐标矩阵
        self.crossover_ratio = 0.2  # 父代选择比
        self.mutate_ratio = 0.05  # 变异概率

        # fruits中存每一个个体是下标的list
        self.dis_mat = self.calculate_dis_mat(num_city, data)  # 计算城市间的路径长度，即路径矩阵
        #self.fruits = self.random_init(num_total, num_city)  # 随机初始化开始种群
        self.fruits = aco_init #直接将蚁群算法得到的解作为遗传算法的初始种群
        # print(len(self.fruits))

        # 显示初始化后的最佳路径
        fitnesses = self.calculate_fitness(self.fruits)
        sort_index = np.argsort(-fitnesses)  # 返回的是种群个体适应值从大到小排序后的索引值的数组
        init_best = self.fruits[sort_index[0]]  # 初始化种群中适应值最高的个体的适应值
        init_best = self.location[init_best]  # 适应值最高的路径的沿途城市坐标

    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

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

    # 计算路径长度
    def calculate_pathlen(self, path, dis_mat):
        try:
            a = path[0]
            b = path[-1]
        except:
            import pdb
            pdb.set_trace()
        result = dis_mat[a][b]
        # print('路径：' , path)
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算种群个体适应度
    def calculate_fitness(self, fruits):
        fitness = []
        for fruit in fruits:
            # if isinstance(fruit, int):
            #     import pdb
            #     pdb.set_trace()
            length = self.calculate_pathlen(fruit, self.dis_mat)
            fitness.append(1.0 / length)
        # print('种群适应度： ', fitness)
        return np.array(fitness)

    # 基因交叉产生下一代
    def ga_cross(self, x, y):
        # print('x' , x , 'y' , y)
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        # print('path_list' , path_list)
        order = list(random.sample(path_list, 2))
        # print('order' , order)
        order.sort()
        start, end = order

        # 找到冲突点并存下他们的下标,x中存储的是y中的下标,y中存储x与它冲突的下标
        tmp = x[start:end]
        # print(tmp)
        x_conflict_index = []
        for sub in tmp:
            index = y.index(sub)
            if not (index >= start and index < end):  # 如果不在start-end之间,则表示如果直接交换会发生冲突
                x_conflict_index.append(index)

        y_confict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = x.index(sub)
            if not (index >= start and index < end):
                y_confict_index.append(index)

        assert len(x_conflict_index) == len(y_confict_index)
        # print(x_conflict_index , y_confict_index)
        # 交叉
        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp
        # print('x ' , x , 'y' , y)

        # 解决冲突
        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            y[i], x[j] = x[j], y[i]

        assert len(set(x)) == len_ and len(set(y)) == len_
        return list(x), list(y)

    # 选择部分优秀的个体作为父代
    def ga_parent(self, fitnesses, crossover_ratio):
        sort_index = np.argsort(-fitnesses).copy()
        sort_index = sort_index[0:int(crossover_ratio * len(sort_index))]
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.fruits[index])
            parents_score.append(fitnesses[index])
        return parents, parents_score

    # 轮盘赌选择算法选择两个父代。
    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]  # 划分区域比率
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    # 变异
    def ga_mutate(self, gene):
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        # np.random.shuffle(tmp)
        tmp = tmp[::-1]
        gene[start:end] = tmp  # 以首尾反转作为变异
        return list(gene)

    def ga(self):
        # 获得优质父代
        fitnesses = self.calculate_fitness(self.fruits)
        # 选择部分优秀个体作为父代候选集合
        parents, parents_score = self.ga_parent(fitnesses, self.crossover_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        # 新的种群fruits
        fruits = parents.copy()
        # 生成新的种群
        while len(fruits) < self.num_total:
            # 轮盘赌方式对父代进行选择
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            # print('轮盘赌：' , gene_x , gene_y)
            # 交叉
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            # 变异
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)
            x_fitness = 1. / self.calculate_pathlen(gene_x_new, self.dis_mat)
            y_fitness = 1. / self.calculate_pathlen(gene_y_new, self.dis_mat)
            # 将适应度高的放入种群中
            if x_fitness > y_fitness and (not gene_x_new in fruits):
                fruits.append(gene_x_new)
            elif x_fitness <= y_fitness and (not gene_y_new in fruits):
                fruits.append(gene_y_new)

        self.fruits = fruits

        return tmp_best_one, tmp_best_score

    def run(self):
        BEST_LIST = None
        best_score = -math.inf
        self.best_record = []
        for i in range(len(iter_x), self.iteration + 1):
            tmp_best_one, tmp_best_score = self.ga()
            iter_x.append(i)
            iter_y.append(1. / tmp_best_score)
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
                iter_path.append(BEST_LIST)
            self.best_record.append(1. / best_score)
            print(i, 1. / best_score)
        print(1. / best_score)
        return BEST_LIST, self.location[BEST_LIST], 1. / best_score


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
        self.iter_max = 2 #200#迭代最大次数
        self.dis_mat = self.calculate_dis_mat(num_city, self.location)  # 计算城市之间的距离矩阵
        self.Eta = 10. / self.dis_mat  # 启发式函数
        #self.Eta = 1. / self.dis_mat  # 启发式函数
        self.paths = None  # 蚁群中每个个体的长度

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
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.calculate_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 更新信息素
    def update_Tau(self):
        delta_tau = np.zeros([self.num_city, self.num_city])
        #paths = self.compute_paths(self.Table)
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
            self.paths = self.compute_paths(self.Table)
            # 取该蚁群的最优解
            tmp_lenth = min(self.paths)
            tmp_path = self.Table[self.paths.index(tmp_lenth)]
            # 可视化每一次迭代的最优路径，从而绘制出路径变化的整个过程
            # 更新最优解
            if tmp_lenth < best_lenth:
                best_lenth = tmp_lenth
                best_path = tmp_path
                #记录下每次迭代选到的最优路径规划图
                iter_path.append(best_path)
            # 更新信息素
            self.update_Tau()

            # 保存结果
            iter_x.append(cnt)
            iter_y.append(best_lenth)
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

#处理信息读入
data = read_tsp('data/pr439.tsp') #结果：[[1.0，64.0，96.0],[],[]...]
data = np.array(data)#将数据转换为矩阵
data = data[:, 1:]#去除掉第一列

time_start=time.time()  #计算算法运行时间
aco = ACO(num_city=data.shape[0], data=data.copy())
Best_pathIndex, Best_path, Best = aco.run()
print('蚁群算法->遗传初始种群: ' , aco.Table , len(aco.Table)) #50
ga = GA(num_city=data.shape[0], num_total=math.ceil(data.shape[0] * 2), iteration=100, data=data.copy() , aco_init = aco.Table)
Best_pathIndex , Best_path , Best_len = ga.run()
print("路径", Best_pathIndex)
time_end=time.time()    #计算算法运行时间
print('time cost',time_end-time_start,'s')

#绘制收敛曲线图
plt.figure()
plt.plot(iter_x , iter_y)
plt.title('混合蚁群遗传算法收敛曲线')
plt.xlabel("迭代次数")
plt.ylabel("路径长度")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

#绘制规划结果图
plt.figure()
# 打开交互模式
plt.ion()
print('长度', len(iter_path))
for i in range(len(iter_path)):
    # 清除原有图像
    plt.cla()
    plt.scatter(Best_path[:, 0], Best_path[:, 1],color='red')  # 所有点的0坐标做x，1坐标做y ,在plt中标出来
    # 使用annotate函数必备参数绘制注解
    # for index, point in zip(Best_pathIndex, Best_path):
    #     plt.annotate(index + 1, xy=(point[0], point[1]))
    plt.title('混合蚁群遗传算法规划结果')
    plt.xlabel("点横坐标")
    plt.ylabel("点纵坐标")
    #print('iter_path[i]', iter_path[i])
    iter_path_coord = aco.location[iter_path[i]]
    iter_path_coord = np.vstack([iter_path_coord, iter_path_coord[0]])  # 将尾和初始连起来
    plt.plot(iter_path_coord[:, 0], iter_path_coord[:, 1])  # 将所有点按顺序连起来
    plt.savefig(r'D:/.c学习资料/人工智能/人工智能大作业/GA_ACOgif\{}.png'.format(i))
    # 暂停
    plt.pause(0.1)
# 关闭交互模式
plt.ioff()
plt.show()

