import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import EarlyStopping
from tkinter import *
from PIL import ImageTk,Image

import matplotlib.pyplot as plt
plt.style.use('ggplot')


##############################################################################################
from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
from tkinter import messagebox
root=Tk()
root.title("Predictive Maintenance")
root.geometry("1920x1080+0+0")
#root.attributes('-fullscreen', True)
root.iconbitmap("st.ico")
img=ImageTk.PhotoImage(Image.open("st.png"))
my_img=Label(image=img)
my_img.place(x=0,y=0)

heading=Label(root,text="Predictive Maintenance using Vibration Analyser",font=("arial",40,"bold"),fg="steelblue").pack()
label1=Label(root,text="Enter Machine id",font=("arial",20,"bold"),fg="black").place(x=10,y=200)
steps=Label(root,text="Steps: \n1.Import the text file\n2.Enter Machine id\n3.Click on Check.",font=("arial",20),fg="black",justify=LEFT).place(x=800,y=200)
m_id=StringVar()
entry_box=Entry(root,textvariable=m_id,width=30,bg="lightblue").place(x=280,y=210)
root.filename="PM_test.txt"
def opFile():
    root.filename = filedialog.askopenfilename(initialdir="E:/Internship/PredictiveMaintenance-2", title="Select A TXT File", filetypes=(("txt files", "*.txt"),("all files", "*.*")))
    print(root.filename)
button1=Button(root,text="Import",width=20,height=3,bg="lightblue",command=opFile).place(x=200,y=300)

##############################################################################################


dataset_train=pd.read_csv('PM_train.txt',sep=' ',header=None).drop([26,27],axis=1)
col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
dataset_train.columns=col_names

dataset_test=pd.read_csv(root.filename,sep=' ',header=None).drop([26,27],axis=1)
dataset_test.columns=col_names

pm_truth=pd.read_csv('PM_truth.txt',sep=' ',header=None).drop([1],axis=1)
pm_truth.columns=['more']
pm_truth['id']=pm_truth.index+1

# generate column max for test data
rul = pd.DataFrame(dataset_test.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']

# run to failure
pm_truth['rtf']=pm_truth['more']+rul['max']

pm_truth.drop('more', axis=1, inplace=True)
dataset_test=dataset_test.merge(pm_truth,on=['id'],how='left')
dataset_test['ttf']=dataset_test['rtf'] - dataset_test['cycle']
dataset_test.drop('rtf', axis=1, inplace=True)

dataset_train['ttf'] = dataset_train.groupby(['id'])['cycle'].transform(max)-dataset_train['cycle']


df_train=dataset_train.copy()
df_test=dataset_test.copy()
period=30
df_train['label_bc'] = df_train['ttf'].apply(lambda x: 1 if x <= period else 0)
df_test['label_bc'] = df_test['ttf'].apply(lambda x: 1 if x <= period else 0)

features_col_name=['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                   's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
target_col_name='label_bc'

sc=MinMaxScaler()
df_train[features_col_name]=sc.fit_transform(df_train[features_col_name])
df_test[features_col_name]=sc.transform(df_test[features_col_name])

def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    id_df=df_zeros.append(id_df,ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)

# function to generate labels
def gen_label(id_df, seq_length, seq_cols,label):
    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    id_df=df_zeros.append(id_df,ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label=[]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label][stop])
    return np.array(y_label)

# timestamp or window size
seq_length=50
seq_cols=features_col_name

# generate X_train
X_train=np.concatenate(list(list(gen_sequence(df_train[df_train['id']==id], seq_length, seq_cols)) for id in df_train['id'].unique()))

# generate y_train
y_train=np.concatenate(list(list(gen_label(df_train[df_train['id']==id], 50, seq_cols,'label_bc')) for id in df_train['id'].unique()))

# generate X_test
X_test=np.concatenate(list(list(gen_sequence(df_test[df_test['id']==id], seq_length, seq_cols)) for id in df_test['id'].unique()))

# generate y_test
y_test=np.concatenate(list(list(gen_label(df_test[df_test['id']==id], 50, seq_cols,'label_bc')) for id in df_test['id'].unique()))

model = tensorflow.keras.models.load_model('pm.h5')
def prob_failure():
    machine_id=m_id.get()
    if(machine_id==""):
        messagebox.showerror("Error", "Please enter a Machine id!")
    if(int(machine_id)>100):
        messagebox.showerror("Error", "Please enter a valid Machine id!")

    machine_df=df_test[df_test.id==int(machine_id)]
    machine_test=gen_sequence(machine_df,seq_length,seq_cols)
    m_pred=model.predict(machine_test)
    failure_prob=list(m_pred[-1]*100)[0]
    label3 = Label(root, text= '\nProbability that Machine {} will fail in 30 days is: {}'.format(machine_id,failure_prob),font=("arial",15),fg="black").place(x=100,y=360)
    print('Probability that machine will fail within 30 days: ',failure_prob)
    return failure_prob

#machine_id=5
#print('Probability that machine will fail within 30 days: ',prob_failure(machine_id))
button2=Button(root,text="Check",width=20,height=3,bg="lightblue",command=prob_failure).place(x=400,y=300)
root.mainloop()


