# Predictive Maintenance with Tkinter GUI

### System Requirements:
* Python 2.7 or above
* Libraries Used:
  * 1)Tensorflow 
  * 2)keras
  * 3)Tkinter
  * 4)pandas
  * 5)numpy
  * 6)scipy
  * 7)sklearn
  * 8)PIL
  * 9)matplotlib
  
  

### PredictiveMaintenance-1
* Dataset Used: Bearing Dataset (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
* Algorithm Used:* 1)Logistic Regression for prediction
                 * 2)Fast Fourier Transform for plotting Graph
                
* File Description:
  * 1.LogisticRegressionTraining.py : Used for training and saving the model
  * 2.FFT_LogiRegress.py : Used to run the model with GUI,also it generates Averaged_BearingTest_Dataset.csv
  * 3.logisticRegressionModel.npy : Logistic Regression Model
  * 4.st.ico , st.png : logos
  * 5.Graph Folder: All the graphs gets stored automatically in this folder 
  * 6.Bearing Dataset : Download the dataset from the above link and unzip it in this directory

* How to run?:
  * 1.First Train the model using LogisticRegressionTraining.py, it will save the model in the same directory i.e logisticRegressionModel.npy (Change file paths inside the code for    training on custom dataset)
  * 2.Run FFT_LogiRegress.py to get the GUI, it contains 3 buttons.
      1.Import: used to select the folder for the testing data
      2.Check: Passes the data to the model and displays the output
      3.Display Graph: Used to Display and save the graph
   
### PredictiveMaintenance-2
* Dataset Used: Turbofan Engine Degradation Simulation Dataset (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
* Algorithm Used: LSTM

* File Description:
  * 1.Train_LSTM_Model.py : Used to train and save the model 
  * 2.PredictiveMaintenance.py : Used to run the model with GUI
  * 3.pm.h5 : LSTM Model
  * 4.PM_train.txt : Data used for training
  * 5.PM_truth.txt : Truth table for training the model
  * 6.PM_test.txt : Data used for testing
  * 7.st.ico , st.png : logos
  * 8.Engine Dataset : Dataset
  
  
