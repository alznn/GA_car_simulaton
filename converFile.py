import pickle
from DataPoint import *

def writeFile(car_log):
    #train4D.txt格式:前方距離、右方距離、左方距離、方向盤得出角度(右轉為正)
    f = open('.\\success_copy\\4DSuccess_train4D.txt','w',encoding='utf-8')
    for index in range(len(car_log.f_dist)):
        f.write(str(car_log.f_dist[index])+" ")
        f.write(str(car_log.r_dist[index])+" ")
        f.write(str(car_log.l_dist[index])+" ")
        f.write(str(car_log.theta[index])+"\n")
        # f.write('\n')
    f.close()

    f = open('.\\success_copy\\4DSuccess_train6D.txt', 'w', encoding='utf-8')
    for index in range(len(car_log.f_dist)):
        f.write(str(car_log.x[index]) + " ")
        f.write(str(car_log.y[index]) + " ")
        f.write(str(car_log.f_dist[index]) + " ")
        f.write(str(car_log.r_dist[index]) + " ")
        f.write(str(car_log.l_dist[index]) + " ")
        f.write(str(car_log.theta[index]) + "\n")
        # f.write('\n')
    f.close()

def Train4D_para():
    with open('.\\train_data\\train4D_gene_200_400.pkl','rb') as f:
        best_gene = pickle.load(f)
    data =[]
    f = open('.\\train_data\\train4D_para.txt','w',encoding='utf-8')
    f.write(str(best_gene.theta)+'\n')
    for index in range(best_gene.rbf_units):
        data = []
        data.append(str(best_gene.weights[index]))
        data.extend([str(m) for m in best_gene.mean[index:index+3]])
        data.append(str(best_gene.dev[index]))
        line = " ".join(data)
        f.write(str(line)+"\n")
    f.close()
def Train6D_para():
    # with open('.\\train_data\\train6D_gene_400_300.pkl','rb') as f:
    with open('350_best_gene.pkl','rb') as f:
        best_gene = pickle.load(f)
    data =[]
    f = open('train6D_para_2.txt','w',encoding='utf-8')
    f.write(str(best_gene.theta)+'\n')
    print(len((best_gene.weights)))
    print(len((best_gene.dev)))
    print(len((best_gene.dev)))
    for index in range(best_gene.rbf_units):
        data = []
        data.append(str(best_gene.weights[index]))
        data.extend([str(m) for m in best_gene.mean[index:index+5]])
        data.append(str(best_gene.dev[index]))
        print(data)
        line = "\t".join(data)
        f.write(str(line)+"\n")
    f.close()
# Train4D_para()
# Train6D_para()
# best_gene = Gene()
# car = car_state()
with open('.\\success_copy\\4D_data_point.pkl','rb') as f:
    car_log = pickle.load(f)
writeFile(car_log)
    # f.write(str(car_log.r_dist[index])+" ")
    # f.write(str(car_log.l_dist[index])+" ")
    # f.write(str(car_log.theta[index])+"\n")