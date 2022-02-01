Python 3.9.6 (tags/v3.9.6:db3ff76, Jun 28 2021, 15:26:21) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import matplotlib.pyplot as plt

>>> import numpy as np
>>> import scipy
>>> import pandas as pd
>>> df = pd.read_csv(r'C:\Users\Deathnote 2\Desktop\Training h5 json data\fer2013\fer2013.csv')

>>> df.head()
   emotion                                             pixels     Usage
0        0  70 80 82 72 58 58 60 63 54 58 60 48 89 115 121...  Training
1        0  151 150 147 155 148 133 111 140 170 174 182 15...  Training
2        2  231 212 156 164 174 138 161 173 182 200 106 38...  Training
3        4  24 32 36 30 32 23 19 20 30 41 21 22 32 34 21 1...  Training
4        6  4 0 0 0 0 0 0 0 0 0 0 0 3 15 23 28 48 50 58 84...  Training
>>> X_train = []
>>> y_train = []
>>> X_test = []
>>> y_test = []
>>> for index, row in df.iterrows():
	k = row['pixels'].split(" ")
	if row['Usage'] == 'Training':
		X_train.append(np.array(k))
		y_train.append(row['emotion'])
	elif row['Usage'] == 'PublicTest':
		X_test.append(np.array(k))
		y_test.append(row['emotion'])

		
>>> import keras
>>> from keras.utils import to_categorical
Traceback (most recent call last):
  File "<pyshell#20>", line 1, in <module>
    from keras.utils import to_categorical
ImportError: cannot import name 'to_categorical' from 'keras.utils' (C:\Users\Deathnote 2\AppData\Roaming\Python\Python39\site-packages\keras\utils\__init__.py)
>>> from tensorflow.keras.utils import to_categorical
>>> y_train= to_categorical(y_train, num_classes=7)
>>> y_test = to_categorical(y_test, num_classes=7)
>>> X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
Traceback (most recent call last):
  File "<pyshell#24>", line 1, in <module>
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
AttributeError: 'list' object has no attribute 'reshape'
>>> X_train = X_train.numpy.reshape(X_train.shape[0], 48, 48, 1)
Traceback (most recent call last):
  File "<pyshell#25>", line 1, in <module>
    X_train = X_train.numpy.reshape(X_train.shape[0], 48, 48, 1)
AttributeError: 'list' object has no attribute 'numpy'
>>> X_train = reshape(X_train.shape[0], 48, 48, 1)
Traceback (most recent call last):
  File "<pyshell#26>", line 1, in <module>
    X_train = reshape(X_train.shape[0], 48, 48, 1)
NameError: name 'reshape' is not defined
>>> X_train = X_train.np.reshape(X_train.np.shape[0], 48, 48, 1)
Traceback (most recent call last):
  File "<pyshell#27>", line 1, in <module>
    X_train = X_train.np.reshape(X_train.np.shape[0], 48, 48, 1)
AttributeError: 'list' object has no attribute 'np'
>>> X_train = np.asarray(X_train)
>>> X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
>>> X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
Traceback (most recent call last):
  File "<pyshell#30>", line 1, in <module>
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
AttributeError: 'list' object has no attribute 'reshape'
>>> X_test = np.asarray(X_test)
>>> X_train = np.asarray(X_train)
>>> X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
>>> from keras.preprocessing.image import ImageDataGenerator
>>> datagen = ImageDataGenerator( 
    rescale=1./255,
    rotation_range = 10,
    horizontal_flip = True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode = 'nearest')
>>> testgen = ImageDataGenerator(rescale=1./255)
>>> datagen.fit(X_train)
>>> batch_size = 64
>>> train_flow = datagen.flow(X_train, y_train, batch_size=batch_size)
>>> from keras.utils import plot_model
Traceback (most recent call last):
  File "<pyshell#40>", line 1, in <module>
    from keras.utils import plot_model
ImportError: cannot import name 'plot_model' from 'keras.utils' (C:\Users\Deathnote 2\AppData\Roaming\Python\Python39\site-packages\keras\utils\__init__.py)
>>> from keras.utils.vis_utils import plot_model
>>> from keras.models import Model
>>> from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
>>> from keras.layers.convolutional import Conv2D
>>> from keras.layers.pooling import MaxPooling2D
>>> from keras.layers.merge import concatenate
>>> from keras.optimizers import Adam, SGD
Traceback (most recent call last):
  File "<pyshell#47>", line 1, in <module>
    from keras.optimizers import Adam, SGD
ImportError: cannot import name 'Adam' from 'keras.optimizers' (C:\Users\Deathnote 2\AppData\Roaming\Python\Python39\site-packages\keras\optimizers.py)
>>> from tensorflow.keras.optimizers import Adam
>>> from keras.regularizers import l1, l2
>>> from matplotlib import pyplot as plt
>>> from sklearn.metrics import confusion_matrix
>>> def FER_Model(input_shape=(48,48,1)):
	# first input model
	visible = Input(shape=input_shape, name='input')
	num_classes = 7
	#the 1-st block
	conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_1')(visible)
	conv1_1 = BatchNormalization()(conv1_1)
	conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_2')(conv1_1)
	conv1_2 = BatchNormalization()(conv1_2)
	pool1_1 = MaxPooling2D(pool_size=(2,2), name = 'pool1_1')(conv1_2)
	drop1_1 = Dropout(0.3, name = 'drop1_1')(pool1_1)
	#the 2-nd block
	conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_1')(drop1_1)
        conv2_1 = BatchNormalization()(conv2_1)
	conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_2')(conv2_1)
	conv2_2 = BatchNormalization()(conv2_2)
	conv2_3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_3')(conv2_2)
	conv2_2 = BatchNormalization()(conv2_3)
	pool2_1 = MaxPooling2D(pool_size=(2,2), name = 'pool2_1')(conv2_3)
	drop2_1 = Dropout(0.3, name = 'drop2_1')(pool2_1)
	#the 3-rd block
	conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_1')(drop2_1)
	conv3_1 = BatchNormalization()(conv3_1)
	conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_2')(conv3_1)
	conv3_2 = BatchNormalization()(conv3_2)
	conv3_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_3')(conv3_2)
	conv3_3 = BatchNormalization()(conv3_3)
	conv3_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_4')(conv3_3)
	conv3_4 = BatchNormalization()(conv3_4)
	pool3_1 = MaxPooling2D(pool_size=(2,2), name = 'pool3_1')(conv3_4)
	drop3_1 = Dropout(0.3, name = 'drop3_1')(pool3_1)
	#the 4-th block
	conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_1')(drop3_1)
	conv4_1 = BatchNormalization()(conv4_1)
	conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_2')(conv4_1)
	conv4_2 = BatchNormalization()(conv4_2)
	conv4_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_3')(conv4_2)
	conv4_3 = BatchNormalization()(conv4_3)
	conv4_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_4')(conv4_3)
	conv4_4 = BatchNormalization()(conv4_4)
	pool4_1 = MaxPooling2D(pool_size=(2,2), name = 'pool4_1')(conv4_4)
	drop4_1 = Dropout(0.3, name = 'drop4_1')(pool4_1)
    
	#the 5-th block
	conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_1')(drop4_1)
	conv5_1 = BatchNormalization()(conv5_1)
	conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_2')(conv5_1)
	conv5_2 = BatchNormalization()(conv5_2)
	conv5_3 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_3')(conv5_2)
	conv5_3 = BatchNormalization()(conv5_3)
	conv5_4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_4')(conv5_3)
	conv5_3 = BatchNormalization()(conv5_3)
	pool5_1 = MaxPooling2D(pool_size=(2,2), name = 'pool5_1')(conv5_4)
	drop5_1 = Dropout(0.3, name = 'drop5_1')(pool5_1)#Flatten and output
	flatten = Flatten(name = 'flatten')(drop5_1)
	ouput = Dense(num_classes, activation='softmax', name = 'output')(flatten)# create model 
	model = Model(inputs =visible, outputs = ouput)
	
SyntaxError: inconsistent use of tabs and spaces in indentation
>>> def FER_Model(input_shape=(48,48,1)):
    # first input model
    visible = Input(shape=input_shape, name='input')
    num_classes = 7
    #the 1-st block
    conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_1')(visible)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2,2), name = 'pool1_1')(conv1_2)
    drop1_1 = Dropout(0.3, name = 'drop1_1')(pool1_1)
    #the 2-nd block
    conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_1')(drop1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_3')(conv2_2)
    conv2_2 = BatchNormalization()(conv2_3)
    pool2_1 = MaxPooling2D(pool_size=(2,2), name = 'pool2_1')(conv2_3)
    drop2_1 = Dropout(0.3, name = 'drop2_1')(pool2_1)
    #the 3-rd block
    conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_1')(drop2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_3')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_4')(conv3_3)
    conv3_4 = BatchNormalization()(conv3_4)
    pool3_1 = MaxPooling2D(pool_size=(2,2), name = 'pool3_1')(conv3_4)
    drop3_1 = Dropout(0.3, name = 'drop3_1')(pool3_1)
    #the 4-th block
    conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_1')(drop3_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_2')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_3')(conv4_2)
    conv4_3 = BatchNormalization()(conv4_3)
    conv4_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_4')(conv4_3)
    conv4_4 = BatchNormalization()(conv4_4)
    pool4_1 = MaxPooling2D(pool_size=(2,2), name = 'pool4_1')(conv4_4)
    drop4_1 = Dropout(0.3, name = 'drop4_1')(pool4_1)
    
    #the 5-th block
    conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_1')(drop4_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_2')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_3 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_3')(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    conv5_4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name = 'conv5_4')(conv5_3)
    conv5_3 = BatchNormalization()(conv5_3)
    pool5_1 = MaxPooling2D(pool_size=(2,2), name = 'pool5_1')(conv5_4)
    drop5_1 = Dropout(0.3, name = 'drop5_1')(pool5_1)#Flatten and output
    flatten = Flatten(name = 'flatten')(drop5_1)
    ouput = Dense(num_classes, activation='softmax', name = 'output')(flatten)# create model 
    model = Model(inputs =visible, outputs = ouput)
    # summary layers
    print(model.summary())
    
    return model

>>> 