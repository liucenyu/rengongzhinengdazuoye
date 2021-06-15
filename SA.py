import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time

class SA(object):
    def __init__(self, num_city, data):
        self.T0 = 100000000000 #初始温度
        self.Tend = 1e-3 #停止温度
        self.rate = 0.99999 #衰减参数-温度的更新
        self.num_city = num_city #城市数量
        self.scores = []
        self.location = data
        # fires中存每一个个体是下标的list
        self.fires = []
        self.dis_mat = self.calculate_dis_mat(num_city, data)
        #self.fire = self.greedy_init(self.dis_mat,100,num_city)
        self.fire = self.random_init(num_city)
        #print(self.fire) [8, 9, 6, 5, 14, 13, 12, 11, 15, 0, 7, 3, 1, 2, 4, 10]
        # 显示初始化后的路径
        init_pathlen = 1. / self.calculate_pathlen(self.fire, self.dis_mat)
        init_best = self.location[self.fire]
        # 存储存储每个温度下的最终路径，画出收敛图
        self.iter_x = [0]
        self.iter_y = [1. / init_pathlen]

    # 初始化一条随机路径
    def random_init(self, num_city):
        tmp = [x for x in range(1,num_city)]
        random.shuffle(tmp)
        tmp = tmp[:] + [0]
        return tmp

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
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算一个温度下产生的一个群体的长度
    def calculate_paths(self, paths):
        result = []
        for one in paths:
            length = self.calculate_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 产生一个新的解：随机交换两个元素的位置
    def get_new_fire(self, fire):
        fire = fire.copy()
        t = [x for x in range(len(fire))]
        a, b = np.random.choice(t, 2)
        fire[a:b] = fire[a:b][::-1]
        return fire

    # 退火策略，根据温度变化有一定概率接受差的解
    def eval_fire(self, raw, get, temp):
        len1 = self.calculate_pathlen(raw, self.dis_mat)
        len2 = self.calculate_pathlen(get, self.dis_mat)
        dc = len2 - len1
        p = max(1e-1, np.exp(-dc / temp))
        if len2 < len1:
            return get, len2
        elif np.random.rand() <= p:
            return get, len2
        else:
            return raw, len1

    # 模拟退火总流程
    def sa(self):
        count = 0
        # 记录最优解
        best_path = self.fire
        best_length = self.calculate_pathlen(self.fire, self.dis_mat)

        while self.T0 > self.Tend:
            count += 1
            # 产生在这个温度下的随机解
            tmp_new = self.get_new_fire(self.fire.copy())
            # 根据温度判断是否选择这个解
            self.fire, file_len = self.eval_fire(best_path, tmp_new, self.T0)
            # 更新最优解
            if file_len < best_length:
                best_length = file_len
                best_path = self.fire
            # 降低温度
            self.T0 *= self.rate
            # 记录路径收敛曲线
            self.iter_x.append(count)
            self.iter_y.append(best_length)
            print(count, best_length)
        return best_length, best_path

    def run(self):
        best_length, best_path = self.sa()
        print('路径：' , best_path)  # 打印路径
        return self.location[best_path], best_length


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

data = np.array(data)
data = data[:, 1:]
show_data = np.vstack([data, data[0]])

time_start=time.time()  #计算算法运行时间
model = SA(num_city=data.shape[0], data=data.copy())
Best_path, Best = model.run()
print(Best)
time_end=time.time()    #计算算法运行时间
print('time cost',time_end-time_start,'s')

# 加上一行因为会回到起点
Best_path = np.vstack([Best_path, Best_path[0]])
fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
axs[0].scatter(Best_path[:, 0], Best_path[:,1])
Best_path = np.vstack([Best_path, Best_path[0]])
#print(Best_path)
axs[0].plot(Best_path[:, 0], Best_path[:, 1]) #画出路径
axs[0].set_title('规划结果')
iterations = model.iter_x
best_record = model.iter_y
axs[1].plot(iterations, best_record)
axs[1].set_title('收敛曲线')
plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签
plt.rcParams['axes.unicode_minus']=False
plt.show()

