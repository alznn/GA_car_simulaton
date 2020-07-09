import math
from DataPoint import *
import shapely.geometry as sp
from copy import deepcopy
import pickle


def read_training(file = 'train4D.txt'):
    input_dim = 0
    train_data = trainset4D()
    print("current file:",file)
    if file == "train4D.txt":
        print("if is 4")
        input_dim = 3
        train_file = open(file,'r',encoding='utf-8')
        # print("lens:",len(train_file.readlines()))
        # print(f.readlines())

        for line in train_file.readlines():
            # line = f.readlines()[index]
            # print("line: ",line)
            # input()
            value = line.split(' ')
            train_data.forward.append(float(value[0]))
            train_data.right.append(float(value[1]))
            train_data.left.append(float(value[2]))
            train_data.output_theta.append(float(value[3]))
        max_forward = max(train_data.forward)
        min_forward = min(train_data.forward)
        max_right = max(train_data.right)
        min_right = min(train_data.right)
        max_left = max(train_data.left)
        min_left = min(train_data.left)
        x_max = max([max_forward, max_right, max_left])
        x_min = min([min_forward, min_right, min_left])
        return input_dim, x_max, x_min, train_data

    elif file=="train6D.txt":
        print("if is 6")
        input_dim = 5
        train_data = trainset6D()
        f = open(file,'r',encoding='utf-8')
        for line in f.readlines():
            value = line.split(' ')
            train_data.X.append(float(value[0]))
            train_data.Y.append(float(value[1]))
            train_data.forward.append(float(value[2]))
            train_data.right.append(float(value[3]))
            train_data.left.append(float(value[4]))
            train_data.output_theta.append(float(value[5]))
        max_x = max(train_data.X)
        min_x = min(train_data.X)
        max_Y = max(train_data.Y)
        min_Y = min(train_data.Y)
        max_forward = max(train_data.forward)
        min_forward = min(train_data.forward)
        max_right = max(train_data.right)
        min_right = min(train_data.right)
        max_left = max(train_data.left)
        min_left = min(train_data.left)
        x_max = max([max_x, max_Y, max_forward, max_right, max_left])
        x_min = min([min_x,min_Y,min_forward, min_right, min_left])

        return input_dim, x_max, x_min, train_data


    # if file_name == 'train6D.txt':

def RBF_Gaussian (x,mean,deviation):
    # print("x: ",x)
    # print("mean: ",mean)
    # input()
    delta_xm = np.array(x)-np.array(mean)
    # print("delta: ",delta_xm)
    # print("(delta_xm.dot(delta_xm)): ",(delta_xm.dot(delta_xm)))
    # print(type(delta_xm))
    # print(deviation)
    output = math.exp(-(delta_xm.dot(delta_xm))/(2 * deviation *deviation))
    # input()
    return output
    # return math.exp(-(x - mean) ** 2 / deviation ** 2)
def RBF(input_x, units, uniq_gene):
    output = uniq_gene.theta
    index = 0
    for i in range(units):
        # print("j: ",i)
        current_means = uniq_gene.mean[index:index+len(input_x)]
            #x, mean, deviation
        tau = RBF_Gaussian(input_x, current_means, uniq_gene.dev[i])
        # print("index: ",index)
        # print("index+len(input): ",index+len(input_x))
        # print("current_means:",current_means)
        # print("uniq_gene.weights[i]: ",uniq_gene.weights[i])
        # print("tau: ",tau)
        output = output + uniq_gene.weights[i] * tau
        # print("uniq_gene.weights[j] * tau: ", uniq_gene.weights[i] * tau)
        # f_x = f_x + self.populations[idx].weight[j] * gaussian
        index+=len(input_x)
    return output
def adaptaive_func(units,inputdim, uniq_gene, train_data):
    #en = sum(output-F(x))^2/2
    en = 0
    err = 0
    train_x = []
    if inputdim == 3:
        for i in range(len(train_data.output_theta)):
            train_x.append([train_data.forward[i],train_data.right[i],train_data.left[i]])
    elif inputdim == 5:
        for i in range(len(train_data.output_theta)):
            train_x.append([train_data.X[i],train_data.Y[i],train_data.forward[i],train_data.right[i],train_data.left[i]])
    # print("len train_x: ",len(train_x))
    # print("train_x: ",train_x[0])
    for index in range(len(train_data.output_theta)):
        input_x = train_x[index]
        rbf = RBF(input_x, units, uniq_gene)
        # if rbf < 0:
        #     print("rbf:",rbf)
        #     input()
        # print('index:', index)
        # print('inpux:', input_x)
        # print('expect:',train_data.output_theta[index])
        # print('rbf:',rbf)
        # denor_rbf = rbf*(max_y-min_y)+min_y #反正規化

        rbfn_value = max(-40, min(rbf * 40, 40))
        # rbfn_value = max(-40, min(rbf, 40))
        # print("after rbfn_value: ", rbfn_value)
        # input()
        delta = train_data.output_theta[index] - rbfn_value
        en += delta*delta
        err+=abs(delta)
        # en += (train_data.output_theta[index] - rbfn_value)
        # en += (train_data.output_theta[index] - rbf)*(train_data.output_theta[index] - rbf)
        # print("e_n", en)
    # 1/2*sum(E_n)
    # input()
    en = en/2 #避免浮點數誤差

    # en = en/len(train_data.output_theta)
    # print("en: ",en)
    # print("en/2: ",en/2)
    # print("1/en: ",1/en)
    error = err/len(train_data.output_theta)
    return 1/en,error

def reproduction(population_num,population,select_way):
    # print("avg_value: ",avg_value)
    pool = []
    # way = 1
    copy_count  = 0
    total = 0
    if select_way == 0: #輪盤式
        # print("#輪盤式")
        adaptive = np.array([p.adative_num for p in population])
        adaptive_ratio = adaptive / np.sum(adaptive)
        indexList = np.random.choice(population_num, population_num, p=adaptive_ratio,replace=False)

        population = [deepcopy(population[idx]) for idx in indexList]
        pool.extend(population)
    else:#競爭式
        for _ in range(population_num):
            indexList = np.random.choice(population_num, 5, p=None, replace=True)
            print("indexList:", indexList)
            temp = [deepcopy(population[idx]) for idx in indexList]
            value_list = [p.adative_num for p in temp]
            # print("temp:", [p.adative_num for p in temp])
            # print("temp:", value_list)
            idx = value_list.index(max(value_list))
            pool.extend([deepcopy(temp[idx])])
    return pool

def mycrossover(parent1,parent2):
    #x1_new = x1+rand_sigma*(x1-x2)  x2 = x2 - rand_sigma*(x1-x2)
    #or
    #x1_new = x1+rand_sigma*(x2-x1)  x2 = x2 - rand_sigma*(x2-x1)
    #random_sigma隨機選取的正實數
    rand_sigma = random.uniform(0,2)
    # print("random_sigma隨機選取的正實數: ",rand_sigma)
    new_p1 = deepcopy(parent1)
    new_p2 = deepcopy(parent2)

    new_p1.theta = parent1.theta + rand_sigma*(parent1.theta-parent2.theta)
    new_p2.theta = parent2.theta - rand_sigma*(parent1.theta-parent2.theta)
    # print("parent1.theta",parent1.theta)
    # print("parent2.theta",parent2.theta)
    # print("new_p1.theta",new_p1.theta)
    # print("new_p2.theta",new_p2.theta)
    # input()
    # parent1.weights = parent1.weights
    # parent2.weights = parent2.weights
    new_p1.weights = parent1.weights + rand_sigma*(parent1.weights-parent2.weights)
    new_p2.weights = parent2.weights - rand_sigma*(parent1.weights-parent2.weights)

    new_p1.mean = parent1.mean + rand_sigma*(parent1.mean-parent2.mean)
    new_p2.mean = parent2.mean - rand_sigma*(parent1.mean-parent2.mean)

    new_p1.dev = parent1.dev + rand_sigma*(parent1.dev-parent2.dev)
    new_p2.dev = parent2.dev - rand_sigma*(parent1.dev-parent2.dev)

    # new_p1.isCrossover = True
    # new_p1.adative_num = None
    # new_p2.isCrossover = True
    # new_p2.adative_num = None

    # new_p1.parent1, new_p1.parent2 = parent1, parent2
    # new_p2.parent1, new_p2.parent2 = parent1, parent2
    return new_p1,new_p2

def select_and_crossover(gene_pool, rate, max_num, min_num,dev):
    #real-code
    # print("=================開始交配!==========================")
    # print("rate: ",rate)

    # print("check select_and_crossover weights.mean ", gene_pool[0].mean)
    # print("check select_and_crossover weights: ", gene_pool[0].weights)
    # print("check select_and_crossover dev: ", gene_pool[0].dev)

    pool = deepcopy(gene_pool)
    if len(pool)%2 == 0:
        for i in range(0,len(pool),2):
            prob = random.uniform(0, 1)
            # prob = 1
            # print("prob: ",prob)
            if prob > rate: #交配
                # print("開始交配!!!!!!!!!!!!")
            # if prob > -1: #交配
            #     print("check type:",type(pool[i]))
            #     input()
                pool[i], pool[i + 1]= mycrossover(pool[i],pool[i+1])
            # pool.append(pool[i])
            # pool.append(pool[i+1])

    else:
        for i in range(0, len(pool)-1, 2):
            prob = random.uniform(0, 1)
            # prob = 1
            # print("prob: ", prob)
            if prob > rate: #交配
                # print("開始交配!!!!!!!!!!!!")
            # if prob > -1:  # 交配
            #     print("check type:", type(pool[i]))
            #     input()
                pool[i], pool[i + 1] = mycrossover(pool[i], pool[i + 1])

    # print("check new pool:", [g.theta for g in pool])
    # print("~~~~~~~~~~~after crossover: \n",pool[0].weights )
    # print("~~~~~~~~~~~after crossover: \n",pool[1].weights )
    # print("~~~~~~~~~~~after crossover: \n",pool[2].weights )
    # print("~~~~~~~~~~~after crossover: \n",pool[-3].weights )
    # print("~~~~~~~~~~~after crossover: \n",pool[-2].weights )
    # print("~~~~~~~~~~~after crossover: \n",pool[-1].weights )
    #data clipping
    for g in pool:
        # normalized = (x - min(x)) / (max(x) - min(x))

        g.theta = np.clip(g.theta,-1,1)
        # g.weights = g.weights
        g.weights = np.clip(g.weights,-1,1)
        g.mean = np.clip(g.mean,min_num,max_num)
        g.dev = np.clip(g.dev, dev/1000, None)
        # g.dev = np.clip(g.dev, 0.0001, 1)
    # print("aaa check new pool:", [g.theta for g in pool])
    return pool
    #mutation(crossover_populations,mutation_rate,num_unit,iteration,population_num,dev, x_down,x_upper)
def mutation(dim,crossover_populations,mutation_rate,num_unit,iteration,population_num,dev, min_num, max_num):
    #x = x+s*noise  s控制加入雜訊之大小
    # print("~~~~~~~~~~~before crossover_populations len: ",len(crossover_populations))
    # print("~~~~~~~~~~~before crossover_populations: ",crossover_populations[0].check())
    mutation_population = []
    # s = random.uniform(1,2)
    # print("len: ",crossover_populations)
    # print("max_num: ",max_num)
    # print("min_num: ",min_num)

    # print("check mutation - crossover_populations mean: ",crossover_populations[0].mean)
    # print("check mutation - crossover_populations weights: ",crossover_populations[0].weights)
    # print("check mutation - crossover_populations dev: ",crossover_populations[0].dev)

    for g in crossover_populations:
        # print("~~~~~~~~~~~before mutation: ",g.check())
        #37.44100656666896, 6.019714470461537
                #Gene(num_unit,input_dim,x_upper, x_down,dev)
        # print(max_num,min_num)
        noise = Gene(units=num_unit, input_dim=dim, x_max=max_num, x_min=min_num, dev_max = dev)
        # print("weight shape",noise.weights.shape)
        # print("noise mean shape",noise.mean.shape)
        # print("g mean shape",g.mean.shape)
        prob = random.uniform(0, 1)
        s = random.uniform(-1, 1)
        if prob > mutation_rate:
            # print("mutation occure")
            # prob = random.uniform(0, 1)
            g.theta += np.random.choice([1, -1]) * s * noise.theta
            g.weights += np.random.choice([1, -1]) * s * noise.weights
            g.mean += np.random.choice([1, -1]) * s * noise.mean
            g.dev += np.random.choice([1, -1]) * s * noise.dev
            g.isMutation = True
            g.adative_num = None
        mutation_population.append(g)
        # print("self.reproduction_pool[i].theta: ", g.theta)
        # print("self.mutation_tiny_adjust_var:", self.mutation_tiny_adjust_var)
        # print(" mutation_factor.theta:", noise.theta)
        # input()
    # data clipping
    # print("~~~~~~~~~~~after mutation: \n",mutation_population[0].mean )
    # print("~~~~~~~~~~~after mutation: \n",mutation_population[0].weights )
    # print("~~~~~~~~~~~after mutation: \n",mutation_population[1].weights )
    # print("~~~~~~~~~~~after mutation: \n",mutation_population[2].weights )
    # input()
    for g in mutation_population:
        # print("~~~~~~~~~~~after mutation: \n", g.check()) use check show none
        g.theta = np.clip(g.theta, -1, 1)
        # g.weights = g.weights
        g.weights = np.clip(g.weights, -1, 1)
        g.mean = np.clip(g.mean, min_num, max_num)
        g.dev = np.clip(g.dev,dev/1000,None)
        # g.dev = np.clip(g.dev,0.0001,1)
        # print("~~~~~~~~~~~after mutation: \n", g.theta)
    # print("aaa check new pool:", [g.theta for g in mutation_population])
    # print("len mutation_population: " , len(mutation_population))
    # input()
    return mutation_population

#迭代次數	 族群大小 突變機率 交配機率 選擇訓練資料集 Save/Load model params
#GA_compute(num_unit=6,iteration = 200,population_num = 400,mutation_rate = 0.6 , crossover_rate = 0.6,dev = 10,file = 'tran4D.txt',model_name = 'c14d.txt')
def GA_compute(num_unit=6,iteration = 3,population_num = 3,mutation_rate = 0.2 , crossover_rate = 0.5,way=0,dev = 10,file = 'train4D.txt',model_name = 'c14d.txt'):
    input_dim, x_upper, x_down, traindata = read_training(file)
    print("input_dim: ",input_dim)
    print("file: ",file)
    print("len traindata: ",len(traindata.output_theta))
    print("traindata: ",traindata.check())
    print("max ans min:", x_upper, x_down, )
    print("mutation_rate: ",mutation_rate)
    print("crossover_rate: ",crossover_rate)
    print("population_num: ",population_num)
    print("iteration: ",iteration)
    print("dev: ",dev)
    print("way: ",way)
    default_populations = []
    best_gene = None
    for i in range(population_num):
        #gene(units=3,input_dim=3,x_max=100,x_min=0)
        tmp_gene = Gene(num_unit, input_dim, x_upper, x_down, dev)
        # print("tmp_gene means shape:",tmp_gene.mean.shape)
        # input()
        default_populations.append(tmp_gene)
    # print(len(default_populations))
    # print(default_populations[0].check())
    # default_populations = sorted(default_populations, key=lambda p: p.adative_num, reverse=True)
    best_gene = deepcopy(default_populations[0])
    best_gene.adative_num, best_gene.error_rate = adaptaive_func(num_unit,input_dim, default_populations[0], traindata)#0.033867102963247186
    populations = default_populations

    print("=======================================start iteration======================================")
    for count in range(iteration):
        print("===================== iter: ",count," ==================================")
        # print("current populations: ", [p.adative_num for p in default_populations])
        #計算自適應函數
        total_value = 0
        # print("len population: ",len(populations))
        # print("current populations weights: ", populations[0].weights)
        for g in populations:
            #adaptaive_func(uniq_gene,units,train_data, flag = 1):
            # print("g:",g.check())
                                                        #(units,inputdim, uniq_gene, train_data)
            g.adative_num, g.error_rate = adaptaive_func(num_unit,input_dim, g, traindata)
            # print("adative_num_value: ", g.adative_num)
            total_value+=g.adative_num
            if g.adative_num > best_gene.adative_num:
                best_gene = deepcopy(g)
                # best_gene = g
                # print("current best_gene: ",best_gene.adative_num)
            # input()
        # print("top 3 best weights: ",[sorted_populations[i].weights for i in range(3)],sep='\n')
        # print("top 3 best mean: ",[sorted_populations[i].mean for i in range(3)],sep='\n')
        # print("top 3 best dev: ",[sorted_populations[i].dev for i in range(3)],sep='\n')
        # print("all adative_num: ",([g.adative_num for g in populations]),sep='\n')
        print("best adative_num: ",max([g.adative_num for g in populations]),sep='\n')
        # print("besr adative_num: ",index(max([g.adative_num for g in populations])),sep='\n')
        # print("all error_rate: ",sorted([g.error_rate for g in populations]),sep='\n')
        # print("min error_rate: ",min([g.error_rate for g in populations]),sep='\n')
        # print("min error_rate: ",index(min([g.error_rate for g in populations])),sep='\n')
        print("\n")
        # print("best_gene.mean :", best_gene.mean)
        print("best_gene.adative_num :", best_gene.adative_num)
        print("best_gene.error_rate :", best_gene.error_rate)
        if count%50==0:
            with open(str(count)+"_best_gene.pkl", 'wb') as f:
                pickle.dump(best_gene, f)


        # print("current populations weights: ",[p.weights for p in populations])
        # print("current populations adative_num: ",[p.adative_num for p in populations])
        # print("best_gene: ",best_gene.adative_num)
        # input()

        # print("avg:", avg_value)
        # print("**********check population adative_num",[g.adative_num for g in populations])
        # print("best_gene adative_num: ",best_gene.adative_num)
        # print(count, 1 /best_gene.adative_num, 1 / avg_value)
        # input()
        #複製
        reproduction_pool = reproduction(population_num, populations, select_way=way)

        print("1 reproduction: ", reproduction_pool[0].weights)
        print("1 len reproduction: ", len(reproduction_pool))
        # print("1 check reproduction weights: ", [cg.weights for cg in reproduction_pool])
        # print("1 check reproduction adative_num: ", [cg.adative_num for cg in reproduction_pool])
        # print(
        #     "**************************************************************************************************************c")
        # input()

        crossover_populations = select_and_crossover(reproduction_pool,crossover_rate, x_upper, x_down,dev)
        # print("2 crossover_populations: ", reproduction_pool[0].weights)
        # print("2 *len crossover: ",len(crossover_populations))
        # print("2 check crossover weights: ",[cg.weights for cg in crossover_populations])
        # print("2 check crossover adative_num: ",[cg.adative_num for cg in crossover_populations])
        # print("**************************************************************************************************************c")
        # input()

        populations = mutation(input_dim,crossover_populations, mutation_rate, num_unit, iteration, population_num,dev, x_down,x_upper)
        # print("3 check mutation weights:",populations[0].weights)
        # print("3 check mutation weights:",[m.weights for m in new_population])
        # print("3 check mutation adative_num:",[m.adative_num for m in new_population])
        # print("**********check mutation weight:",[m.weights for m in new_population])
        # input()

        # print("current populations weights: ", populations[0].weights)
        # print("current populations weights: ", [p.weights for p in populations])
        # print("current populations adative_num: ", [p.adative_num for p in populations])
        # print(
        #     "===================================================================================================================")
    print("!!!!!!!!!!!!!!!!!!!!final!!!!!!!!!!!!!!!!!!!!")
    print("check best adative_num: ", best_gene.adative_num)
    print("check best error rate: ", best_gene.error_rate)
    print("file: ",file)
    #start interation
    postfix = "_".join(["_gene", str(iteration), str(population_num)])
    print("postfix: ", postfix)
    para_file = file.replace('.txt', postfix + '.pkl')
    # train4D_gene_200_400.pkl
    print("para file", para_file)
    with open(para_file,'wb') as f:
        pickle.dump(best_gene,f)
    return best_gene
# GA_compute()