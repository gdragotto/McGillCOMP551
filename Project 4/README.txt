Caveat: the directory needs to be changed to the image folder; and if you also want to use comet, 
you need to create an account on comet, and change the corresponding arguments for comet.

-------------
Introduction
-------------
Folder "for_paper_reproduction" contains:
1. squeezenet.py
   The pre-trained SqueezeNet model (v1.1), originated from Keras

2. paper_reproduction.py
   Predicting top-1 and top-5 accuracy on ImageNet (ILSVRC 2012) validation dataset, this script calls SqueezeNet function from squeezenet.py file.
	
3. ValidationLog.csv

   The predicted results on validation set.

Note: The validation set is 6.3GB, which can be downloaded from http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar.


Folder "for_exploration" contains 14 experiments, 1 preprocessing script and the log file for Model9_SN_af4:
Note: All 14 experiments are conducted on Tiny ImageNet dataset (237MB), which can be downloaded from https://tiny-imagenet.herokuapp.com. The preprocessing script needs to run first!!

Preprocessing: val_set_refactor.py - for organizing validation folder in the same structure as the training set.

Experiments:
1. For exploring learning rate:
	1) Model1_SN_lr1.py: learning rate = 0.04
           Other parameters: momentum = 0.9, dropout ratio = 0.5, activation function (all) = ReLU, optimizer = SGD

	2) Model2_SN_lr2.py: learning rate = 0.01

	3) Model3_SN_lr3.py: learning rate = 0.04 * (0.1**(#epoch/30)) 
	   The same learning rate method as the paper

Note: We set Model1_SN_lr1 as our baseline because it exceeds the performance of paper's method (Model3_SN_lr3). 
The following experiments use the same parameters as Model1_SN_lr1, if not specified.

2. For exploring dropout ratio:
	1) Model4_SN_dr1.py: dropout ratio = 0.25

	2) Model5_SN_dr2.py: dropout ratio = 0.75

3. For exploring activation function:
	1) Model6_SN_af1.py: activation function(Conv10) = Softplus, dropout = 0.25

	2) Model7_SN_af2.py: activation function(Conv10) = Softplus, dropout = 0.75

	3) Model8_SN_af3.py: activation function(Feature) = LeakyReLU

	4) Model9_SN_af4.py: activation function(all) = LeakyReLU

4. For exploring optimizer:
	1) Model10_SN_op1: optimizer = Adam
	   Here, we used the same learning rate and momentum as those of SGD in Model1_SN_lr1, and set epsilon as 0.1, betas as default in PyTorch.

	2) Model11_SN_op2: optimizer = Adam, activation function(all) = LeakyReLU

5. For exploring model components on macroarchitecture level
	1) Model12_SN_f1: removing fire9 module

	2) Model13_SN_m1: adding maxpooling layer between fire7 and fire8 modules (dropout = 0.25)

	3) Model14_SN_re1: adding ResNet bypass connections around fire3, fire5, fire7 and fire9 (dropout = 0.25)
	   Setting learning rate = 0.06, momentum = 0.95 (for fastening accuracy improvement)

Log file: Model9_SN_af4_log.txt is the log file for our best model - Model9_SN_af4, which achieves the highest top-1 and top-5 accuracy within our experiments.

----------------------------
Credits to Python Libraries
----------------------------
For reproducing results in SqueezeNet paper:
numpy
keras

For exploration on model components:
torch
torchvision
comet_ml (for generating an interface to track code, experiments and results)

(The detailed packages are shown in the scripts)

Note: We changed from Keras to PyTorch when exploring model components because it's easier to make changes to parameters and model layers on the SqueezeNet script from PyTorch!!!


------------------------------------
Approximate runtime on GPU platform
------------------------------------
Most of the experiments are conducted on a same machine GPU type - NVIDIA TITAN Xp, two of them (Model9 and Model10) on NVIDIA P100-PCIE. All of them were used 2 GPUs.
Most models need ~4 hours with 2 NVIDIA TITAN Xp GPUs. 
Some models need to be manually stopped because they are not able to achieve the early_stopping condition.

Somehow experiments that run on NVIDIA P100-PCIE need much more time, but by conducting similar experiments on these two types of GPUs, 
we are certain that Model9 and Model10 only need ~4 hours with 2 NVIDIA TITAN Xp GPUs.





