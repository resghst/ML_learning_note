
#in[1]
from keras.utils import np_utils
import numpy as np
np.random.seed(10)
#in[2]
from keras.datasets import mnist
(x_train_image,y_train_label),\
(x_test_image,y_test_label)= mnist.load_data()
#in[3]
x_Train =x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
#in[4]
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255
#in[5]
y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)
#in[6]
from keras.models import Sequential
from keras.layers import Dense
#in[7]
model = Sequential()
#in[8]
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
#in[9]
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
#in[10]
print(model.summary())
#in[11]
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
#in[12]
train_history =model.fit(x=x_Train_normalize,
                         y=y_Train_OneHot,validation_split=0.2, 
                         epochs=10, batch_size=200,verbose=2)
#in[13]
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
#in[14]
show_train_history(train_history,'acc','val_acc')
#in[15]
show_train_history(train_history,'loss','val_loss')
#in[16]
scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print()
print('accuracy=',scores[1])
#in[17]
prediction=model.predict_classes(x_Test)
#in[18]
print(prediction)
#in[19]
import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,
                                  idx,num=10):
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
#in[20]
plot_images_labels_prediction(x_test_image,y_test_label,
                              prediction,idx=0)
#in[21]
import pandas as pd
pd.crosstab(y_test_label,prediction,
            rownames=['label'],colnames=['predict'])
#in[22]
df = pd.DataFrame({'label':y_test_label, 'predict':prediction})
print(df[:2])
#in[23]
df[(df.label==5)&(df.predict==3)]
#in[24]
plot_images_labels_prediction(x_test_image,y_test_label
                              ,prediction,idx=340,num=1)
#in[25]
plot_images_labels_prediction(x_test_image,y_test_label
                              ,prediction,idx=1289,num=1)