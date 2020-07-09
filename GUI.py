from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from draw import draw_moving_car,draw_map
from car_move import *
from DataPoint import *
import os
import pickle

track_dir = '.\\track_data'
train_dir = '.\\weights'

# from kernal import get_gui_setting,drawpicture
class skin():
    def __init__(self):
        self.org_canvas = Canvas(window, width=600, height=600)
        self.img = PhotoImage(file='')
        self.imgArea = self.org_canvas.create_image(0, 0, anchor=NW, image=self.img)
        self.org_canvas.place(x=480, y=10, anchor='nw')

        self.track_case = ttk.Combobox(window,
                                         values=["case01.txt"], font=('Arial', 10))
        self.track_case.place(x=10, y=10)
        self.track_case.current(0)

        self.train_data = ttk.Combobox(window,
                                         values=["train4D.txt", "train6D.txt"], font=('Arial', 10))
        self.train_data.place(x=10, y=40)
        self.train_data.current(0)

        self.reproduction_way = ttk.Combobox(window,
                                         values=["輪盤式選擇","競爭式選擇"], font=('Arial', 10))
        self.reproduction_way.place(x=10, y=70)
        self.reproduction_way.current(0)
        Label(window, text='神經元：數量: ', font=('Arial', 12)).place(x=10, y=100)
        Label(window, text='請輸入正整數，表示RBF神經元數量: ', font=('Arial', 10)).place(x=10, y=130)
        Label(window, text='迭代次數: ', font=('Arial', 12)).place(x=10, y=180)
        Label(window, text='請輸入正整數，程式將在抵達迭代次數後停止: ', font=('Arial', 10)).place(x=10, y=210)
        Label(window, text='族群大小:', font=('Arial', 12)).place(x=10, y=260)
        Label(window, text='基因數量，請輸入正整數，數量越大則執行越慢:', font=('Arial', 10)).place(x=10, y=290)
        Label(window, text='突變機率:', font=('Arial', 12)).place(x=10, y=340)
        Label(window, text='突變機率，範圍於 0 到 1 之間，數值越大交配機率越低:', font=('Arial', 10)).place(x=10, y=370)
        Label(window, text='交配機率： ', font=('Arial', 12)).place(x=10, y=420)
        Label(window, text='交配機率，範圍介於 0 到 1 之間，數值越大交配機率越高 ', font=('Arial', 10)).place(x=10, y=450)

        unit = StringVar()
        unit.set('6')
        unit = Entry(window, textvariable=unit, font=('Arial', 10))
        unit.place(x=130, y=100)

        iterrations = StringVar()
        iterrations.set('200')
        iterrations = Entry(window, textvariable=iterrations, font=('Arial', 10))
        iterrations.place(x=100, y=180)

        population = StringVar()
        population.set('400')
        population = Entry(window, textvariable=population, font=('Arial', 10))
        population.place(x=100, y=260)

        mutation = StringVar()
        mutation.set('0.2')
        mutation = Entry(window, textvariable=mutation, font=('Arial', 10))
        mutation.place(x=100, y=340)

        crossover= StringVar()
        crossover.set('0.5')
        crossover = Entry(window, textvariable=crossover, font=('Arial', 10))
        crossover.place(x=100, y=420)

        self.btn_train = Button(window, text='train param', command=lambda:train_model()).place(x=10, y=500)
        self.show = Button(window, text='show param', command=lambda:get_para()).place(x=130, y=500)
        self.default = Button(window, text='default_success', command=lambda: success()).place(x=10, y=530)

        my_string_var = StringVar(value="Default Value")

        my_label = Label(window, textvariable=my_string_var,justify=LEFT,padx=10, font=('Arial', 8))
        my_label.place(x=10, y=580)
        def success():
            my_string_var.set("Load parameters!!!")

            para = GA_Setting()
            para.iteration = int(iterrations.get())
            para.population_num = int(population.get())
            para.crossover_rate = float(crossover.get())
            para.mutation_rate = float(mutation.get())
            para.reprduction_way = self.reproduction_way.current()

            select_file = self.track_case.get()
            select_train_file = self.train_data.get()

            track_file = os.path.join(track_dir, select_file)
            train_file = os.path.join(train_dir, select_train_file)

            print("Load parameter!")
            success_file=''
            car_point = ''
            if self.train_data.current()==0:
                car_point = '.\\success_copy\\4D_data_point.pkl'
                success_file = '.\\success_copy\\train4D_gene_200_400.pkl'
            elif self.train_data.current()==1:
                success_file = '.\\success_copy\\train6D_para_2.pkl'
                car_point = '.\\success_copy\\6D_data_point_2.pkl'

            with open(success_file, 'rb') as f:
                assigned_para = pickle.load(f)

            show_info = '\n'.join([":".join(["File", str(car_point.replace('.\\train_data\\','').replace('_point.pkl',''))]),
                                   ":".join(["Node", str(assigned_para.rbf_units)]),
                                   ":".join(["error rate: ", str(assigned_para.error_rate)]),
                                   ":".join(["theta", str(assigned_para.theta)]),
                                   ":".join(["adative_num: ", str(assigned_para.adative_num)]),
                                   ":".join(["weights: ", str(assigned_para.weights)]),
                                   ":".join(["dev: ", str(assigned_para.dev)]),
                                   ":".join(["mean: ", str(assigned_para.mean)])])
            my_string_var.set(show_info)

            self.get_map()
            #File = '.\\track_data\case01.txt',car_track='data_point.pkl'
            #4D_data_point.pkl
            draw_moving_car(track_file,car_point)
        def train_model():
            my_string_var.set("train GA!,please wait at list 1 hour")

            para = GA_Setting()
            para.units = int(unit.get())
            print("units: ",para.units)
            para.iteration = int(iterrations.get())
            para.population_num = int(population.get())
            para.crossover_rate = float(crossover.get())
            para.mutation_rate = float(mutation.get())
            para.reprduction_way = self.reproduction_way.current()

            select_file = self.track_case.get()
            select_train_file = self.train_data.get()
            track_file = os.path.join(track_dir, select_file)
            train_file = os.path.join(train_dir, select_train_file)
            self.get_map()

            is_success = main_run(para=para, File=track_file,Train_file = self.train_data.get() ,file_ID = self.train_data.current())
            if is_success:
                my_string_var.set("success!!!click show button to see result")
                messagebox.showinfo(title='result', message='"success!!!! "')
            else:
                my_string_var.set("Failed, please try again!!!")
                messagebox.showinfo(title='result', message='"碰！ Σヽ(ﾟД ﾟ; )ﾉ "')
            # draw_moving_car(file)

            car_point = ''
            if self.train_data.current()==0:
                car_point = '4D_data_point.pkl'
            elif self.train_data.current()==1:
                car_point = '6D_data_point.pkl'
            draw_moving_car(track_file,car_point)

        def get_para():
            my_string_var.set("Load parameters!!!")

            para = GA_Setting()
            para.iteration = int(iterrations.get())
            para.population_num = int(population.get())
            para.crossover_rate = float(crossover.get())
            para.mutation_rate = float(mutation.get())
            para.reprduction_way = self.reproduction_way.current()

            select_file = self.track_case.get()
            select_train_file = self.train_data.get()

            track_file = os.path.join(track_dir, select_file)

            print("Load parameter!")
            postfix = "_".join(["_gene", str(para.iteration), str(para.population_num)])
            print("postfix: ", postfix)
            para_file = select_train_file.replace('.txt', postfix + '.pkl')
            # train4D_gene_200_400.pkl
            print("para file", para_file)
            para_file = os.path.join(train_dir, para_file)
            assigned_para = Gene()

            if FileNotFoundError:
                my_string_var.set('\nFileNotFoundError, Train First!')

            with open(para_file, 'rb') as f:
                assigned_para = pickle.load(f)

            show_info = '\n'.join([":".join(["theta", str(assigned_para.theta)]),
                                   ":".join(["adative_num: ", str(assigned_para.adative_num)]),
                                   ":".join(["weights: ", str(assigned_para.weights)]),
                                   ":".join(["dev: ", str(assigned_para.dev)]),
                                   ":".join(["mean: ", str(assigned_para.mean)])])
            my_string_var.set(show_info)

            self.get_map()
            #File = '.\\track_data\case01.txt',car_track='data_point.pkl'
            #4D_data_point.pkl
            car_point = ''
            if self.train_data.current()==0:
                car_point = '4D_data_point.pkl'
            elif self.train_data.current()==1:
                car_point = '6D_data_point.pkl'
            draw_moving_car(track_file,car_point)
    def get_map(self):
        select_file = self.track_case.get()
        select_train_file = self.train_data.get()
        track_file = os.path.join(track_dir, select_file)
        train_file = os.path.join(train_dir, select_train_file)

        print("track_file: ",track_file)
        print("train_file: ",train_file)
        # draw_map(track_file)
        file_name = track_file.replace('.\\track_data\\', '').replace('.txt', '.png')
        from PIL import Image
        # type+"_"+file+".png")
        # file_name = str(self.comboExample.get()).replace('.txt', '.png')
        im = Image.open(file_name)
        print(im.size[0])
        print(im.size[1])
        nim = im.resize((70*9,65*9), Image.BILINEAR)
        nim.save(file_name)

        self.img = PhotoImage(file=file_name)
        self.org_canvas.itemconfig(self.imgArea, image=self.img)


#
# 第1步，例項化object，建立視窗window
window = Tk()
# 第2步，給視窗的視覺化起名字
window.title('My Window')
# 第3步，設定視窗的大小(長 * 寬)
window.geometry('1200x1000')  # 這裡的乘是小x
# 第4步，載入 wellcome image
# file = [data for data in os.listdir(dir)]
app = skin()
window.mainloop()