import math
import matplotlib.pyplot as plt
import numpy as np

## 动态规划法
class DP(object):
    def __init__(self, num_city, data):#70 , 坐标矩阵
        self.num_city = num_city
        self.location = data
        self.dis_mat = self.calculate_dis_mat(num_city, data) #dis_mat说白了就是一张二维距离表

    # 计算不同城市之间的距离
    def calculate_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = 0 #到自己的距离设置为0
                    continue
                #获取两个点的坐标
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)])) #计算两点之间的距离
                dis_mat[i][j] = tmp
        # for i in dis_mat:
        #     for j in i:
        #         print(j , end=' ')
        #     print()
        return dis_mat

    #计算路径长度, goback:是否计算回到起始点的路径
    def calculate_pathlen(self, path, dis_mat, goback=True):
        try:
            a = path[0] # 0
            b = path[-1] # 1
        except:
            import pdb
            pdb.set_trace()
        if goback:
            result = dis_mat[a][b] #加上最终回到a点的距离
        else:
            result = 0.0
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b] #加上邻两点之间的距离
        return result

    def run(self):
        tmppath = [0] #从0经过所有点且仅经过一次走到0
        tmplen = 0
        # self.location 为距离矩阵
        print(self.dis_mat)
        cnt = 1 << (self.num_city - 1)
        dp = [[float(0) for col in range(cnt)] for row in range(self.num_city)]
        dp = np.array(dp)
        #初始化dp
        for i in range(self.num_city):
            dp[i][0] = self.dis_mat[i][0]
        print(dp)

        #以列从左往右计算dp数组 dp[v][S]表示从v经过S中所有节点各1次到达0的最短距离，所以目标函数为dp[0][cnt-1]
        for S in range(1 , cnt):
            #遍历所有顶点v
            for v in range(self.num_city):
                dp[v][S] = np.inf
                #print(dp[v][S])
                #当S中包含v时跳过
                if ( v != 0 and ((S >> (v - 1)) & 1) == 1):
                    continue
                #遍历所有顶点，若S中包含点k，则例举v到k，再经过S / k到0的距离
                for k in range(1,self.num_city):
                    if (1 << (k-1)) & S == 0: continue
                    #print(dp[k][S ^ (1 << (k - 1))] + self.dis_mat[v][k])
                    dp[v][S] = min(dp[v][S], dp[k][S ^ (1 << (k - 1))] + self.dis_mat[v][k])
                    #print('dp[v][S]: ', v , S , dp[v][S])
        tmplen=dp[0][cnt - 1]
        print('动态规划距离为: ' ,tmplen)

        S = cnt - 1
        v = 0
        #print(1 , end = ' ')
        while S:
            for k in range(1 , self.num_city):
                if (1 << (k-1)) & S == 0: continue
                if dp[v][S] == dp[k][S ^ (1 << (k - 1))] + self.dis_mat[v][k]:
                    tmppath = tmppath[:] + [k]
                    S = S ^ (1 << (k - 1))
                    v = k
                    break
        tmppath = tmppath[:] + [0]
        print('tmppath = ' , tmppath)
        return tmppath, self.location[tmppath], tmplen

# 读取数据
def read_tsp(path):
    lines = open(path, 'r').readlines() #读取文件所有行直到遇到EOF，每一行一个列表项
    assert 'NODE_COORD_SECTION\n' in lines #判断是否读取成功了
    index = lines.index('NODE_COORD_SECTION\n') #获取‘NODE_COORD_SECTION’所在行号
    data = lines[index + 1:-1] #将冗余信息清除干净

    #测试是否读取、处理成功
    # for line in data:  # 依次读取列表中的内容
    #     line = line.strip() # 去掉每行头尾空白
    #     print(line)

    tmp = []
    for line in data:
        line = line.strip().split(' ') #获取每一行，并且按照‘ ’进行分割
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x)) #转换成float类型加入到tmpline列表中
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp  #每一项是一个三元组
    return data


data = read_tsp('data/ulysses16.tsp') #结果：[[1.0，64.0，96.0],[],[]...]
#print(data)
data = np.array(data)#将数据转换为矩阵
#print(data)
data = data[:, 1:]#去除掉第一列
#print(data)


model = DP(num_city=data.shape[0], data=data.copy())#data.shape[0]第一维的长度，即行有70
Best_pathIndex, Best_path, Best = model.run() #Best_path为最优的路径结果，即顺序点集，Best为其长度
#model.run()
# print(Best_path)
#
print('规划的路径长度:{}'.format(Best))
# # 显示规划结果
plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签
plt.rcParams['axes.unicode_minus']=False

plt.scatter(Best_path[:, 0], Best_path[:, 1]) #所有点的0坐标做x，1坐标做y ,在plt中标出来
# 使用annotate函数必备参数绘制注解
for index,point in zip(Best_pathIndex,Best_path):
    plt.annotate(index+1,xy=(point[0] , point[1]),color='red')
Best_path = np.vstack([Best_path, Best_path[0]]) #将尾和初始连起来
plt.plot(Best_path[:, 0], Best_path[:, 1]) #将所有点按顺序连起来
plt.title('DP规划结果')
plt.xlabel("点横坐标")
plt.ylabel("点纵坐标")
plt.show()

