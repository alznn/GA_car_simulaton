import math
from DataPoint import *
from GA import GA_compute
import shapely.geometry as sp
from copy import deepcopy
import pickle

def writeFile(car_log):
    #train4D.txt格式:前方距離、右方距離、左方距離、方向盤得出角度(右轉為正)
    f = open('6DSuccess_train4D.txt','w',encoding='utf-8')
    for index in range(len(car_log.f_dist)):
        f.write(str(car_log.f_dist[index])+" ")
        f.write(str(car_log.r_dist[index])+" ")
        f.write(str(car_log.l_dist[index])+" ")
        f.write(str(car_log.theta[index])+"\n")
        # f.write('\n')
    f.close()

    f = open('6DSuccess_train6D.txt', 'w', encoding='utf-8')
    for index in range(len(car_log.f_dist)):
        f.write(str(car_log.x[index]) + " ")
        f.write(str(car_log.y[index]) + " ")
        f.write(str(car_log.f_dist[index]) + " ")
        f.write(str(car_log.r_dist[index]) + " ")
        f.write(str(car_log.l_dist[index]) + " ")
        f.write(str(car_log.theta[index]) + "\n")
        # f.write('\n')
    f.close()

def readFile(File):
    print("File:",File)
    car_init = CarInfo()
    track = TrackInfo()
    f = open('.\\track_data\\case01.txt','r')
    lines = []
    for line in f.readlines():
        line = line.replace('\n','')
        lines.append(line)
    for index in range(len(lines)):
        if index == 0:
            car_init.x,car_init.y,car_init.fai\
                = [int(i) for i in lines[index].split(',')]
        else:
            x, y = [int(i) for i in lines[index].split(',')]
            if index==1 or index==2:
                track.insert_end(x,y)
            else:
                track.insert_node(x, y)
    return car_init,track

def RBF_Gaussian (x,mean,deviation):
    # print("x: ",x)
    # print("mean: ",mean)
    # input()
    delta_xm = np.array(x)-np.array(mean)
    output = math.exp(-(delta_xm.dot(delta_xm))/(2 * deviation *deviation))
    return output
    # return math.exp(-(x - mean) ** 2 / deviation ** 2)
def RBF(input_x,units,uniq_gene):
    print("~~~~~~~~~~~~~~~~~~ RBF prediction ~~~~~~~~~~~~~~~~~~")
    output = uniq_gene.theta
    index = 0
    for i in range(units):
        # print("j: ",i)
        current_means = uniq_gene.mean[index:index+len(input_x)]
            #x, mean, deviation
        tau = RBF_Gaussian(input_x,current_means,uniq_gene.dev[i])
        output = output + uniq_gene.weights[i] * tau
        # print("car output: ",output)

        index+=len(input_x)
    return output

#output = rbfn_funct(np.array(list4d), best_parameters)
#def update(x,y,fai,theta,b):
def update(x,y,fai,theta,b):
    print("X:",x)
    new_x = x + math.cos(math.radians(fai +theta))+math.sin(math.radians(fai))*math.sin(math.radians(theta))
    print("new_X:",new_x)
    new_y = y + math.sin(math.radians(fai +theta))-math.sin(math.radians(theta))*math.cos(math.radians(fai))
    print("new_y:",new_y)

    new_fai =fai-math.degrees(math.asin((math.sin(math.radians(theta))*2)/b))
    print("new_fai: ",new_fai)

    return new_x,new_y,new_fai

def main_run(para=None, File='case01.txt',Train_file = 'train6D.txt',file_ID=0):
    # paras = Guassion_Function()
    # paras = para
    # with open('train4D_gene_200_400.pkl','rb') as f:
    # with open('train6D_para_2.pkl','rb') as f:
    #     best_gene = pickle.load(f)
    # file_ID = 0
    if para==None:
        units = 9
        iterations = 400
        population_nums = 300
        crossover_rate = 0.5
        mutation_rate = 0.2
        reprduction_way = 0
        Train_file = 'train6D.txt'
        file_ID=1
        # pass
    else:
        units = para.units
        iterations = para.iteration
        population_nums = para.population_num
        crossover_rate = para.crossover_rate
        mutation_rate = para.mutation_rate
        reprduction_way = para.reprduction_way
        print("file_ID: ",file_ID)
        print("File: ",File)
    best_gene = GA_compute(num_unit=units,iteration = iterations,population_num = population_nums,
                           mutation_rate = mutation_rate , crossover_rate = crossover_rate,way=reprduction_way,dev = 10,file =Train_file,model_name = 'c14d.txt')
    # best_gene = Gene()
    # best_gene.theta = -0.02651955
    # best_gene.mean=np.array([4.6436,4.6436, 4.6436,  4.6436,  4.6436, 40.1364,  4.6436,  4.6436,  4.6436,
    #         4.6436, 40.1364,  4.6436,  4.6436,  4.6436, 40.1364,  4.6436, 40.1364,  4.6436])
    # best_gene.weights=np.array( [-0.13855804 ,- 0.4499985 ,  0.1560278 ,  0.0757076 ,- 0.6667405  , 0.83175379])
    # best_gene.dev= np.array( [17.52630357, 17.39668874, 23.97969245 ,26.60256349, 23.69022832, 23.00550918])
    # best_gene.adative_num =  8.838660122353671
    print("check best adative_num: ", best_gene.adative_num)
    print("check best error_rate: ", best_gene.error_rate)
    print("check best rbf_units: ", best_gene.rbf_units)
    print("check best theta: ", best_gene.theta)
    print("check best weights: ", best_gene.weights)
    print("check best mean: ", best_gene.mean)
    print("check best: len mean", len(best_gene.mean))
    print("check best: dev", best_gene.dev)
    print("check best: ", best_gene.check())

    count = 0
    mv_range = 1
    flag = True
    # CarInfo(x,y,theta,fai)

    # premise = Premise_Set()
    # File_path ='.\data\\'+File
    car_current, track = readFile(File)
    # draw car circle
    carObj = sp.Point(car_current.x, car_current.y).buffer(car_current.r)

    # draw track
    trackObj = sp.LineString([[track.nodes_x[i], track.nodes_y[i]] for i in range(len(track.nodes_y))])
    # print(trackObj)

    # draw endline 長方形
    endPolyObj = sp.Polygon([(track.ends_x[0], track.ends_y[0]),
                             (track.ends_x[1], track.ends_y[0]),
                             (track.ends_x[1], track.ends_y[1]),
                             (track.ends_x[0], track.ends_y[1])])
    print(endPolyObj)
    # input()
    ###初始化
    car = car_state()
    # 4d : forward, left, right, output_theta
    # 6d : x, y, forward, left, right, output_theta
    # traindata = read_training()

    # (x, y, forward, left, right, theta, fai)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!running Start!!!!!!!!!!!!!!!!!!!!!!!!!
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!running Start!!!!!!!!!!!!!!!!!!!!!!!!!')
    while True:
        print(
            '****************************************************************************************************************')
        if (endPolyObj.contains(sp.Point(car_current.x, car_current.y))):
            print("get END")
            # with open("data_point.pkl", 'wb') as f:
            #     pickle.dump(car, f)
            # writeFile(car)
            if file_ID == 0:
                print("4D_data_point is save!")
                with open("4D_data_point.pkl", 'wb') as f:
                    pickle.dump(car, f)
                return 1
            if file_ID == 1:
                # with open("6D_data_point.pkl", 'wb') as f:
                #     print("6D_data_point is save!")
                #     pickle.dump(car, f)
                return 1
            # break
            # pass
        if flag:
            # print("get in flag")
            # (x, y, theta, fia):
            car.insert_carlog(car_current.x, car_current.y, car_current.fai)  # 車子狀態
            car.insert_newtheta(car_current.theta)  # 方向盤狀態
            max_x = max(track.nodes_x)
            max_y = max(track.nodes_y)
            min_x = min(track.nodes_x)
            min_y = min(track.nodes_y)
            # print("Max and Min: ", (max_x, max_y), (min_x, min_y))
            mv_range = math.sqrt(((max_x - min_x) ** 2) + ((max_y - min_y) ** 2))
            # print("moving_range: ",mv_range)
            flag = False
        # elif (carObj.intersection(trackObj)) and flag == False:
        #     print("碰！  \ ( ‵ A ′ )/  ")
        #     break
        else:
            new_x, new_y, new_fai = update(car_current.x, car_current.y, car_current.fai, car_current.theta,
                                           car_current.r * 2)
            car_current.x = new_x
            car_current.y = new_y
            car_current.fai = new_fai
            car.insert_carlog(new_x, new_y, new_fai)
            # car.insert_newtheta(car_current.theta)  # 方向盤狀態
            carObj = sp.Point(car_current.x, car_current.y).buffer(car_current.r)
            print(car_current.x, car_current.y)
            if (carObj.intersection(trackObj)) and flag == False:
                print("碰！  \ ( ‵ A ′ )/  ")
                if file_ID == 0:
                    print("4D_data_point is save!")
                    with open("4D_data_point.pkl", 'wb') as f:
                        pickle.dump(car, f)
                    return 0
                if file_ID == 1:
                    with open("6D_data_point.pkl", 'wb') as f:
                        print("6D_data_point is save!")
                        pickle.dump(car, f)
                    return 0

        setSensor(car, car_current, trackObj, mv_range)
        # print("forward distance: ", car.f_dist[-1])
        # print("right distance: ", car.r_dist[-1])
        # print("l_dist distance: ", car.l_dist[-1])

        # RBF(input_x,units,uniq_gene)
        input_x = []
        if int(len(best_gene.mean)/best_gene.rbf_units) == 3:
            input_x = [car.f_dist[-1], car.r_dist[-1], car.l_dist[-1]]
        elif int(len(best_gene.mean)/best_gene.rbf_units) == 5:
            input_x = [car.x[-1],car.y[-1],car.f_dist[-1], car.r_dist[-1], car.l_dist[-1]]
        # print("input_x: ",input_x)
        # print("input_x: ",len(best_gene.mean))
        # input()
        new_theta = RBF(input_x, best_gene.rbf_units, best_gene)

        new_theta = max(-40, min(new_theta*40, 40))

        print("!!!new theta: !!!",new_theta)

        car_current.theta = new_theta
        car.insert_newtheta(new_theta)

        count += 1
        # input()


def setSensor(car_log, car_current, track, mv_range):
    # 車體中心設有感測器，可偵測正前方與左右各45度之距離
    # 前方與牆的距離
    forward_pt = [[car_current.x, car_current.y],
                  [car_current.x + mv_range * math.cos(math.radians(car_current.fai)),
                   car_current.y + mv_range * math.sin(math.radians(car_current.fai))]]
    f_wall = sp.LineString(forward_pt).intersection(track)
    f_point, f_dist = getDistance(f_wall, car_current)
    # print("f_dist: ", f_dist)
    # input()

    # 左右牆距離，右為正(0-90)，固往右打的角度應該為減
    right_pt = [[car_current.x, car_current.y],
                [car_current.x + mv_range * math.cos(math.radians(car_current.fai - 45)),
                 car_current.y + mv_range * math.sin(math.radians(car_current.fai - 45))]]
    r_wall = sp.LineString(right_pt).intersection(track)
    r_point, r_dist = getDistance(r_wall, car_current)
    # print("r_dist: ", r_dist)
    # print("r_point: ", r_point)
    # input()

    left_pt = [[car_current.x, car_current.y],
               [car_current.x + mv_range * math.cos(math.radians(car_current.fai + 45)),
                car_current.y + mv_range * math.sin(math.radians(car_current.fai + 45))]]
    l_wall = sp.LineString(left_pt).intersection(track)
    l_point, l_dist = getDistance(l_wall, car_current)
    # print("l_dist: ", l_dist)
    # print("l_point: ", l_point)
    # input()
    # insert_sensorlog(self,forward,left,right,f_dist,left_dist,right_dist)
    car_log.insert_sensorlog(f_point, l_point, r_point, f_dist, l_dist, r_dist)


def getDistance(wall, car):
    dist_list = []
    min_dist = 9999999999
    min_point = []
    # print("Car center: ", (car.x,car.y))
    if isinstance(wall, sp.Point):
        dist = math.sqrt(((wall.x - car.x) ** 2
                          + (wall.y - car.y) ** 2))
        if dist < min_dist:
            min_dist = dist
            min_point = [wall.x, wall.y]
        dist_list.append(dist_list)
    elif isinstance(wall, sp.MultiPoint):
        for data in range(0, len(wall)):
            dist = math.sqrt(((wall[data].x - car.x) ** 2 + (wall[data].y - car.y) ** 2))
            if (dist < min_dist):
                min_dist = dist
                min_point = [wall[data].x, wall[data].y]
    return min_point, min_dist

# main_run()