from matplotlib import pyplot as plt
from matplotlib import animation
# from car_move import readFile
from DataPoint import *
from matplotlib.widgets import Button

'''
TrackInfo():
   self.start_point = [-6, -3]
   self.nodes_x = self.getNode()[0] # x list
   self.nodes_y = self.getNode()[1] # y list
   self.ends_x = self.getEndPos()[0] #list
   self.ends_y = self.getEndPos()[1] #list
'''
import pickle

def readFile(File):
    print("draw_read_file\n")
    print("File:",File)
    car_init = CarInfo()
    track = TrackInfo()
    f = open(File,'r')
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

def draw_map(File = '.\\track_data\case01.txt'):
    car_init, mytrack = readFile(File)
    fig, ax = plt.subplots()
    # fig.set_size_inches(7, 6.5)
    ax = plt.axes(xlim=(-50, 80), ylim=(-50, 60))
    plt.plot(mytrack.nodes_x, mytrack.nodes_y, 'k-')
    plt.plot(mytrack.ends_x, mytrack.ends_y, 'brown')
    plt.axis('equal')
    circle1 = plt.Circle((car_init.x, car_init.y), 3, fc='y')  # car
    ax.add_artist(circle1)
    plt.savefig(File.replace('.\\track_data\\','').replace('.txt','.png'))
    plt.show()


def draw_moving_car(File = '.\\track_data\case01.txt',car_track='4D_data_point.pkl'):
    # _,mytrack = readFile(File)
    _,mytrack = readFile( File)
    # with open('data_point.pkl','rb') as f:
    with open(car_track,'rb') as f:
        dataset = pickle.load(f)
    #
    fig, ax = plt.subplots()

    ax = plt.axes(xlim=(-50, 80), ylim=(-50, 60))
    plt.plot(mytrack.nodes_x, mytrack.nodes_y, 'k-')
    plt.plot(mytrack.ends_x, mytrack.ends_y,'brown')
    plt.axis('equal')
    patch = plt.Circle((dataset.x[0], dataset.y[0]), 3, fc='y')  # car

    # forward distance
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = "init"
    forward_text = ax.text(0.05, 1.00, textstr, transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', bbox=props)
    front_line, = ax.plot(dataset.forward[0][0], dataset.forward[0][1], 'red',lw=3)
    # right distance
    rprops = dict(boxstyle='round', facecolor='green', alpha=0.5)
    textstr = "init"
    right_text = ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=rprops)
    right_line, = ax.plot([dataset.x[0], dataset.right[0][1]], [dataset.y[0], dataset.right[0][1]], 'blue')

    # left distance
    lprops = dict(boxstyle='round', facecolor='blue', alpha=0.5)
    textstr = "init"
    left_text = ax.text(0.05, 0.90, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=lprops)
    left_line, = ax.plot([dataset.x[0], dataset.left[0][1]], [dataset.y[0], dataset.left[0][1]], 'green')

    def car_init():
        patch.center = (dataset.x[0], dataset.y[0])
        forward_text.set_text('initial')
        right_text.set_text('initial')
        left_text.set_text('initial')
        # print(dataset.forward[0][0])
        front_line.set_data(dataset.forward[0][0],dataset.forward[0][1])
        left_line.set_data(dataset.left[0][0],dataset.left[0][1])
        left_line.set_data(dataset.left[0][0],dataset.left[0][1])

        ax.add_patch(patch)
        # return front_line,
        return patch, forward_text, right_text,left_text,front_line,right_line,left_line

    def car_animate(i):
        # x, y = patch.center
        x = dataset.x[i]
        y = dataset.y[i]
        patch.center = (x, y)

        forward_text.set_text('forward dist: ' + str(dataset.f_dist[i]))
        right_text.set_text('right dist: ' + str(dataset.r_dist[i]))
        left_text.set_text('left dist: ' + str(dataset.l_dist[i]))

        front_line.set_data([dataset.forward[i][0],x],[dataset.forward[i][1],y])
        # front_line.set_ydata(dataset.forward[i][1])
        right_line.set_data([dataset.right[i][0],x], [dataset.right[i][1],y])
        left_line.set_data([dataset.left[i][0],x], [dataset.left[i][1],y])
        # print(dataset.forward[i][0],"  ",dataset.forward[i][1])
        # print('************************')
        # print(dataset.left[i][0],"  ",dataset.left[i][1])
        # print('============================')
        # print(dataset.right[i][0], "  ", dataset.right[i][1])
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # return front_line,#left_line,
        return patch, forward_text, right_text, left_text,front_line,right_line,left_line

    anim = animation.FuncAnimation(fig, car_animate, init_func=car_init, frames=len(dataset.x),
                                   interval=100, blit=True, repeat=False)
    plt.show()
# draw_moving_car(car_track='4D_data_point.pkl')
# draw_map()
    # imagemagick
