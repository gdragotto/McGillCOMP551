

BASIC SETUP:
All script assumes "train_images.pkl", and "train_labels.csv" are in the current working directory in which the scripts are executed.
model3.py or model3_cpu.py also requires "./test_images.pkl" to be present, as it is our final model used for the Kaggle submission.



TO RUN: 
python [any_script].py



LIBRARY REQUIREMENTS:
sklearn, numpy, pandas, pickle, matplotlib, TensorFlow, Keras, Pytorch



SCRIPTS:

model3.py:
This is an implementation of our residual neural network (ResNN) model. 
This training and validation accuracy are plotted against the number of epochs.
The script used for our final submission on Kaggle, it was run using 2 GPUs. 
In the case where the setup requirements are not satisfied, we have provided a script for cpu machines: "model3_cpu.py"



model3_cpu.py:
Version of the "model1.py" script that runs on cpu device.



model2.py:
This is an implementation of our mlCNN model, and the trained model was saved as "my_model2.h5"
The validation accuracy is reported at the end of training.



preProcess.py:
This is the pre-processing step to try and extract the largest digit from the images using OpenCV.
The processed images are outputted into two files: "trainingSet.npy", and "testSet.npy".
The "trainingSet.npy" is essential for "model1.py" to run.



model1.py:
This is an implementation of our LeNet model. To run this model, we need to first run preProcess.py to generate the processed images by OpenCV.
The file: "trainingSet.npy" is required by this python script. 



model1_cpu.py:
Version of the "model1.py" script that runs on cpu device.






