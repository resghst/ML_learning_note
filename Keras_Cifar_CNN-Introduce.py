#In[1]
import numpy
from keras.datasets import cifar10
import numpy as np
np.random.seed(10)
#In[2]
(x_img_train,y_label_train), \
(x_img_test, y_label_test)=cifar10.load_data()
#In[3]
print('train:',len(x_img_train))
print('test :',len(x_img_test))
#In[4]
print(x_img_train.shape)
#In[5]
print(y_label_train.shape)
#In[6]
print(x_img_test.shape)
#In[7]
print(x_img_test[0])
#In[8]
print(y_label_test.shape)
#In[9]
label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
            5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
#In[10]
import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,
                                  idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
                
        title=str(i)+','+label_dict[labels[i][0]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[i]]
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()
#In[11]
plot_images_labels_prediction(x_img_train,y_label_train,[],0)
#In[12]
print('x_img_test:',x_img_test.shape)
print('y_label_test :',y_label_test.shape)
#In[13]
print(x_img_train[0][0][0])
#In[14]
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
#In[15]
print(x_img_train_normalize[0][0][0])
#In[16]
print(y_label_train.shape)
#In[17]
print(y_label_train[:5])
#In[18]
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
#In[19]
print(y_label_train_OneHot.shape)
#In[20]
print(y_label_train_OneHot[:5])
#In[21]

#In[22]

#In[23]

#In[24]

#In[25]

#In[26]

#In[27]

#In[28]

#In[29]

#In[30]

#In[31]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]

#In[]
