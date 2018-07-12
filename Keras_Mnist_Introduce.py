#in[1]
import numpy as np 
import pandas as pd 
from keras.utils import np_utils 
np.random.seed(10)
#in[2]
from keras.datasets import mnist
#in[3]
(x_train_image, y_train_label), \
(x_test_image, y_test_label) = mnist.load_data()
#in[4]
print('train data=',len(x_train_image))
print(' test data=',len(x_test_image))
#in[5]
print ('x_train_image:',x_train_image.shape)
print ('y_train_label:',y_train_label.shape)
#in[6]
import matplotlib.pyplot as plt
def plot_image(image):
	fig = plt.gcf()
	fig.set_size_inches(2, 2)
	plt.imshow(image, cmap='binary')
	plt.show()
#in[7]
# plot_image(x_train_image[0]) # show image by pyplot
#in[8]
print(y_train_label[0])
#in[9]
import matplotlib.pyplot as plt
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
	fig = plt.gcf()
	fig.set_size_inches(12, 14)
	if num>25: num=25 
	for i in range(0, num):
		ax=plt.subplot(5,5, 1+i)
		ax.imshow(images[idx], cmap='binary')
		title= "label=" +str(labels[idx])
		if len(prediction)>0:
			title+=",predict="+str(prediction[idx]) 
				
		ax.set_title(title,fontsize=10) 
		ax.set_xticks([]);ax.set_yticks([])        
		idx+=1 
		plt.show()
#in[10]
# plot_images_labels_prediction(x_train_image,y_train_label,[],0,10)
#in[11]
print ('x_test_image:',x_test_image.shape)
print ('y_test_label:',y_test_label.shape)
#in[12]
# plot_images_labels_prediction(x_test_image,y_test_label,[],0,10)
#in[13]
print ('x_train_image:',x_train_image.shape)
print ('y_train_label:',y_train_label.shape)
#in[14]
x_Train =x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
#in[15]
print ('x_train:',x_Train.shape)
print ('x_test:',x_Test.shape)
#in[16]
# print(x_train_image[0])
#in[17]
x_Train_normalize = x_Train/ 255
x_Test_normalize = x_Test/ 255
#in[18]
# print(x_Train_normalize[0])
#in[19]
print(y_train_label[:5])
#in[20]
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)
#in[21]
print(y_TrainOneHot[:5])


















