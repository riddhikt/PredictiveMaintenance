#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import messagebox
from scipy.fftpack import fft
from scipy.spatial.distance import cdist
from sklearn import cluster
import os
from tkinter import ttk
import tkinter as tk

text = str()
########################### GUI ###########################
root = Tk()
root.title("Predictive Maintenance")
root.geometry("1920x1080+0+0")
# root.attributes('-fullscreen', True)
root.iconbitmap("st.ico")
img = ImageTk.PhotoImage(Image.open("st.png"))
my_img = Label(image=img)
my_img.grid(row=0,column=0)
heading = Label(root, text="Predictive Maintenance using Vibration Analyser", font=("arial", 40, "bold"),
                fg="steelblue")
heading.grid(row=0, column=1,columnspan=6)

filedir = "Bearing Dataset/2nd_test/"
filepath = "Bearing Dataset/2nd_test/"
folder_selected = "Bearing Dataset/2nd_test/"
def opFile():
    # root.withdraw()
    folder_selected = filedialog.askdirectory()
    folder_selected = folder_selected.replace('\\', '/')
    print(folder_selected)
    filedir = folder_selected  # input("enter the complete directory path ")
    filepath = folder_selected

button1 = Button(root, text="Import", width=20, height=3, bg="lightblue", command=opFile)


# user input for the path of the dataset
  # input("enter the folder name")


# cal_labels function take no_of_files as input and generate the label based on 70-30 split.
# files for the testset1 = 2148,testset2 = 984,testset3 = 6324
def cal_Labels(files):
    range_low = files * 0.7
    range_high = files * 1.0
    label = []
    for i in range(0, files):
        if (i < range_low):
            label.append(0)
        elif (i >= range_low and i <= range_high):
            label.append(1)
        else:
            label.append(2)
    return label


# cal_amplitude take the fftdata, n = no of maximun amplitude as input and return the top5 frequecy which has the highest amplitude
def cal_amplitude(fftData, n):
    ifa = []
    ia = []
    amp = abs(fftData[0:int(len(fftData) / 2)])
    freq = np.linspace(0, 10000, num=int(len(fftData) / 2))
    ida = np.array(amp).argsort()[-n:][::-1]
    ia.append([amp[i] for i in ida])
    ifa.append([freq[i] for i in ida])
    return (ifa, ia)


# this function calculate the top n freq which has the heighest amplitude and retuen the list for each maximum
def cal_max_freq(files, path):
    freq_max1, freq_max2, freq_max3, freq_max4, freq_max5 = ([] for _ in range(5))
    for f in files:
        temp = pd.read_csv(path + f, sep="\t", header=None, engine='python')
        temp_freq_max1, temp_freq_max2, temp_freq_max3, temp_freq_max4, temp_freq_max5 = ([] for _ in range(5))
        if (path == "1st_test/"):
            rhigh = 8
        else:
            rhigh = 4
        for i in range(0, rhigh):
            t = fft(temp[i])
            ff, aa = cal_amplitude(t, 5)
            temp_freq_max1.append(np.array(ff)[:, 0])
            temp_freq_max2.append(np.array(ff)[:, 1])
            temp_freq_max3.append(np.array(ff)[:, 2])
            temp_freq_max4.append(np.array(ff)[:, 3])
            temp_freq_max5.append(np.array(ff)[:, 4])
        freq_max1.append(temp_freq_max1)
        freq_max2.append(temp_freq_max2)
        freq_max3.append(temp_freq_max3)
        freq_max4.append(temp_freq_max4)
        freq_max5.append(temp_freq_max5)
    return (freq_max1, freq_max2, freq_max3, freq_max4, freq_max5)


# take the labels for each bearing, plot the corrosponding graph for each bearing .

def plotlabels(labels):
    length = len(labels)
    leng = len(labels[0])
    if (length == 4):
        ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=2, colspan=1)
        ax2 = plt.subplot2grid((8, 1), (2, 0), rowspan=2, colspan=1)
        ax3 = plt.subplot2grid((8, 1), (4, 0), rowspan=2, colspan=1)
        ax4 = plt.subplot2grid((8, 1), (6, 0), rowspan=2, colspan=1)
        y1 = ax1.scatter(np.array(range(1, leng + 1)), np.array(labels)[0], label="bearing1")
        y2 = ax2.scatter(np.array(range(1, leng + 1)), np.array(labels)[1], label="bearing2")
        y3 = ax3.scatter(np.array(range(1, leng + 1)), np.array(labels)[2], label="bearing3")
        y4 = ax4.scatter(np.array(range(1, leng + 1)), np.array(labels)[3], label="bearing4")
        plt.legend(handles=[y1, y2, y3, y4])
    elif (length == 8):
        ax1 = plt.subplot2grid((16, 1), (0, 0), rowspan=2, colspan=1)
        ax2 = plt.subplot2grid((16, 1), (2, 0), rowspan=2, colspan=1)
        ax3 = plt.subplot2grid((16, 1), (4, 0), rowspan=2, colspan=1)
        ax4 = plt.subplot2grid((16, 1), (6, 0), rowspan=2, colspan=1)
        ax5 = plt.subplot2grid((16, 1), (8, 0), rowspan=2, colspan=1)
        ax6 = plt.subplot2grid((16, 1), (10, 0), rowspan=2, colspan=1)
        ax7 = plt.subplot2grid((16, 1), (12, 0), rowspan=2, colspan=1)
        ax8 = plt.subplot2grid((16, 1), (14, 0), rowspan=2, colspan=1)
        y1 = ax1.scatter(np.array(range(1, leng + 1)), np.array(labels)[0], label="bearing1_x")
        y2 = ax2.scatter(np.array(range(1, leng + 1)), np.array(labels)[1], label="bearing1_y")
        y3 = ax3.scatter(np.array(range(1, leng + 1)), np.array(labels)[2], label="bearing2_x")
        y4 = ax4.scatter(np.array(range(1, leng + 1)), np.array(labels)[3], label="bearing2_y")
        y5 = ax5.scatter(np.array(range(1, leng + 1)), np.array(labels)[4], label="bearing3_x")
        y6 = ax6.scatter(np.array(range(1, leng + 1)), np.array(labels)[5], label="bearing3_y")
        y7 = ax7.scatter(np.array(range(1, leng + 1)), np.array(labels)[6], label="bearing4_x")
        y8 = ax8.scatter(np.array(range(1, leng + 1)), np.array(labels)[7], label="bearing4_y")
        plt.show()
        plt.legend(handles=[y1, y2, y3, y4, y5, y6, y7, y8])


def create_dataframe(freq_max1, freq_max2, freq_max3, freq_max4, freq_max5, bearing):
    result = pd.DataFrame()
    result['fmax1'] = list((np.array(freq_max1))[:, bearing])
    result['fmax2'] = list((np.array(freq_max2))[:, bearing])
    result['fmax3'] = list((np.array(freq_max3))[:, bearing])
    result['fmax4'] = list((np.array(freq_max4))[:, bearing])
    result['fmax5'] = list((np.array(freq_max5))[:, bearing])
    x = result[["fmax1", "fmax2", "fmax3", "fmax4", "fmax5"]]
    return x


def elbow_method(X):
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = cluster.KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    #  Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def reshapelist(lst):
    return (list(map(list, zip(*lst))))


def check():

    #if filedir == " " and filepath == " ":
        #messagebox.showerror("Error", "Please import the folder!")
        #opFile()
    # load the files
    all_files = os.listdir(filedir)
    freq_max1, freq_max2, freq_max3, freq_max4, freq_max5 = cal_max_freq(all_files, filepath)
    # load the model
    filename = "logisticRegressionModel.npy"
    logisticRegr = np.load(filename).item()

    # checking the iteration
    if (filepath == "Bearing Dataset/1st_test/"):
        rhigh = 8
    else:
        rhigh = 4

    print("for the testset", filepath)
    prediction_last_100 = []
    j = 3
    k = 10
    l = 20
    for i in range(0, 4):

        currlabel = 'label' + str(j)
        print("Checking for the bearing", i + 1)
        currlabel = Label(root, text='\nChecking for the bearing {}'.format(i + 1),font=("bold"))
        currlabel.grid(row=j, column=1, sticky=W)
        j = j + 1
        currlabel1 = 'label' + str(j)
        # creating  the dataframe
        x = create_dataframe(freq_max1, freq_max2, freq_max3, freq_max4, freq_max5, i)
        predictions = logisticRegr.predict(x)
        prediction_last_100.append(predictions[-100:])
        # count no of zeros
        zero = list(predictions).count(0)
        ones = list(predictions).count(1)
        print("the no of passed files", zero)
        currlabel1 = Label(root, text='The no of passed files {}'.format(zero))
        currlabel1.grid(row=j, column=1, sticky=W)
        j = j + 1
        currlabel2 = 'label' + str(j)
        print("the no of failed files", ones)
        currlabel2 = Label(root,text='The no of failed files {}'.format(ones))
        currlabel2.grid(row=j, column=1, sticky=W)
        j += 1
        check_one = list(predictions[-100:]).count(1)
        check_zero = list(predictions[-100:]).count(0)

        currlabel3 = 'label' + str(j)
        if (check_one > check_zero):
            currlabel3 = Label(root, text="Bearing is suspected, there are chances to fail")
            currlabel3.grid(row=j, column=1, sticky=W)
            j += 1
            print("bearing is suspected, there are chances to fail")
        else:
            currlabel3 = Label(root, text="Bearing has no issue")
            currlabel3.grid(row=j, column=1, sticky=W)
            j += 1
            print("Bearing has no issue")

    # plotting the last 100 prediction for each bearing
    # plotlabels(prediction_last_100)
    #

def fft_graph():

    # if filedir == " " and filepath == " ":
        #messagebox.showerror("Error", "Please import the folder!")
        #opFile()

    merged_data = pd.DataFrame()

    for fname in os.listdir(filedir):
        dataset = pd.read_csv(os.path.join(filedir, fname), sep='\t')
        dataset_mean_abs = np.array(dataset.abs().mean())
        dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1, 4))
        dataset_mean_abs.index = [fname]
        merged_data = merged_data.append(dataset_mean_abs)

    merged_data.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
    # transform data file index to datetime and sort in chronological order
    merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
    merged_data = merged_data.sort_index()
    merged_data.to_csv('Averaged_BearingTest_Dataset.csv')
    print("Dataset shape:", merged_data.shape)
    merged_data.head()
    test = merged_data['2004-02-15 12:52:39':]
    # transforming data from the time domain to the frequency domain using fast Fourier transform
    test_fft = np.fft.fft(test)
    # frequencies of the degrading sensor signal
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(test_fft[:, 0].real, label='Bearing 1', color='blue', animated=True, linewidth=1)
    ax.plot(test_fft[:, 1].imag, label='Bearing 2', color='red', animated=True, linewidth=1)
    ax.plot(test_fft[:, 2].real, label='Bearing 3', color='green', animated=True, linewidth=1)
    ax.plot(test_fft[:, 3].real, label='Bearing 4', color='black', animated=True, linewidth=1)
    plt.legend(loc='lower left')
    ax.set_title('Bearing Sensor Test Frequency Data', fontsize=16)
    #plt.show()
    now= datetime.now()
    current_time = now.strftime("%Y-%m-%d %H-%M-%S")
    plt.savefig('Graphs/'+current_time+'.png')

    novi = Toplevel()
    canvas = Canvas(novi, width=1120, height=480)
    canvas.pack(expand=YES, fill=BOTH)
    gif1 = PhotoImage(file='Graphs/'+current_time+'.png')
    # image not visual
    canvas.create_image(50, 10, image=gif1, anchor=NW)
    # assigned the gif1 to the canvas object
    canvas.gif1 = gif1

    # canvas.get_tk_widget().grid(row=5, column=1)


button1.grid(row=2, column=0)
button2 = Button(root, text="Check", width=20, height=3, bg="lightblue", command=check)
button2.grid(row=2, column=1)
button3 = Button(root, text="Display Graph", width=20, height=3, bg="lightblue", command=fft_graph)
button3.grid(row=2, column=2,sticky=W,padx=30)
root.mainloop()