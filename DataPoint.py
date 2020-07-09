import numpy as np
import random
import cmath
class Gene():
    def __init__(self,units=6,input_dim=3,x_max=100,x_min=0,dev_max = 1):
        self.isCrossover = False
        self.isMutation = False

        self.parent1 = None
        self.parent2 = None
        self.error_rate = 0.0
        self.rbf_units = units
        self.dim = 1+units+units*input_dim+units
        # self.theta = 0.8682546
        self.theta = random.uniform(-1, 1)
        self.weights = self.init_weight(units)
        # self.weights = np.array([0.66619523,0.6551805,0.83271377,-0.89851229,-0.55301806,0.82231405])
        self.mean = self.init_mean(units,input_dim,x_max,x_min)  #unit*input_dim
        # self.mean = np.array([21.11911409,7.68613249,13.45024786,32.56872009,9.47476112,29.623338,23.17621892,30.92474296,6.32569461,13.73441606,12.32420703,37.28822389,40.06198923,6.73717919,13.04993178,37.63070058,19.27587397,5.23177848])
        self.dev = self.init_dev(units,dev_max)
        # self.dev =np.array([8.39099719,8.73639774,8.8572641,5.92638547,6.87148915,7.51157936])
        self.adative_num = None #自適應函數值

    def init_weight(self,units):
        return  np.array([random.uniform(-1, 1) for _ in range(units)])
    def init_mean(self,units,input_dim,x_max,x_min):
    # def init_dist_mean(self,units,input_dim,x_max,x_min,y_max,y_min,f_max,f_min,r_max,r_min,l_max,l_min):
        # means_value = []
        # for _ in range(units):
        #     if input_dim == 3:
        #         f_mean = random.uniform(f_max,f_min)
        #         r_mean = random.uniform(r_max,r_min)
        #         l_mean = random.uniform(l_max,l_min)
        #         means_value.append(f_mean)
        #         means_value.append(r_mean)
        #         means_value.append(l_mean)
        #     if input_dim == 5:
        #         x_mean = random.uniform(x_max,x_min)
        #         y_mean = random.uniform(y_max,y_min)
        #         f_mean = random.uniform(f_max,f_min)
        #         r_mean = random.uniform(r_max,r_min)
        #         l_mean = random.uniform(l_max,l_min)
        #         means_value.append(x_mean)
        #         means_value.append(y_mean)
        #         means_value.append(f_mean)
        #         means_value.append(r_mean)
        #         means_value.append(l_mean)
        # print("units",units)
        # print("input_dim",input_dim)
        return np.array([random.uniform(x_min,x_max) for _ in range(units*input_dim)])
    def init_dev(self,units,dev_max):
        return np.array([random.uniform(dev_max/100, dev_max) for _ in range(units)])

    def check(self):
        print("====================check start=======================")
        print('theta', self.theta)
        print('weight', self.weights)
        print('means', self.mean)
        print('sd', self.dev)
        print('adapt_value', self.adative_num)
        print("====================  check end  =======================")

class trainset4D():
    def __init__(self):
        #前方、右方、左方、方向盤角度(右轉為正)
        self.forward =[]
        self.right = []
        self.left = []
        self.output_theta = []  #方向盤角度
    def check(self):
        print('forward', self.forward[0:3])
        print('right', self.right[0:3])
        print('left', self.left[0:3])
        print('output_theta', self.output_theta[0:3])

class trainset6D():
    def __init__(self):
        #X,Y,前方、右方、左方、方向盤角度(右轉為正)
        self.X = []
        self.Y = []
        self.forward = []
        self.right = []
        self.left = []
        self.output_theta = []  #方向盤角度
    def check(self):
        print('X', self.X[0:3])
        print('Y', self.Y[0:3])
        print('forward', self.forward[0:3])
        print('right', self.right[0:3])
        print('left', self.left[0:3])
        print('output_theta', self.output_theta[0:3])

class TrackInfo():
    def __init__(self):
        self.start_point = [-6, -3]
        self.nodes_x = [] # x list
        self.nodes_y = [] # y list
        self.ends_x = [] #list
        self.ends_y = [] #list
    def insert_end(self,x,y):
        self.ends_x.append(x)
        self.ends_y.append(y)
        # end_x =  [18, 30]
        # end_y =  [37, 40]
        return self.ends_x,self.ends_y
    def insert_node(self, x, y):
        self.nodes_x.append(x)
        self.nodes_y.append(y)
        # node_X = [-6,-6,18,18,30,30,6,6,-6]
        # node_y = [-3,22,22,50,50,10,10,-3,-3]
        return self.nodes_x,self.nodes_y
class CarInfo():
    def __init__(self):
        self.theta = 0.0 #方向盤
        self.x = 0.0
        self.y = 0.0
        self.fai = 0.0 #車子與水平面夾角
        self.r = 3 #直徑為6

class GA_Setting():
    def __init__(self):
        self.units = 6
        self.iteration = 200
        self.population_num = 400
        self.mutation_rate = 0.2
        self.crossover_rate = 0.5
        self.reprduction_way = 1
'''
class Premise_Set():
    def __init__(self):
        self.f_large=0.0
        self.f_medium=0.0
        self.f_small=0.0
        self.lr = 0.0
        self.lr_l=0.0
        self.lr_m=0.0
        self.lr_s=0.0
'''
class car_state():
    def __init__(self):
        self.x=[]
        self.y=[]

        self.fia=[]       #車子角度
        self.theta=[]       #方向盤

        self.forward=[]     #前方牆壁距離座標
        self.left=[]        #左方牆壁距離座標
        self.right=[]       #左方牆壁距離座標

        self.f_dist = []  # 前方牆壁距離變化紀錄
        self.l_dist = []  # 左方牆壁距離變化紀錄
        self.r_dist = []  # 左方牆壁距離變化紀律
    def insert_carlog(self,x,y,fia,):
        self.x.append(x)
        self.y.append(y)
        self.fia.append(fia)
    def insert_sensorlog(self,forward,left,right,f_dist,left_dist,right_dist):
        self.forward.append(forward)    #
        self.left.append(left)
        self.right.append(right)
        self.f_dist.append(f_dist)
        self.l_dist.append(left_dist)
        self.r_dist.append(right_dist)
    def insert_newtheta(self,theta):
        self.theta.append(theta)
# node = Gene()
# print(node.weights)
# print(node.mean)
# print(node.dev)
# print(node.adative_num)