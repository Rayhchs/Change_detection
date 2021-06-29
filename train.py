import numpy as np
import cv2
import os
from model import unet
from keras.callbacks import EarlyStopping

def generate(N_BANDS, BATCH_SIZE, N_DATA, img_sz):
	i = 0
	while 1:
	    xtrain1 = []
	    xtrain2 = []
	    labels = []
	    for b in range(BATCH_SIZE):
	        if i == N_DATA:
	            i = 0
	        i += 1

	        y_label = np.zeros(shape=(img_sz[0],img_sz[1],2),dtype=np.float32)
	        img_1 = cv2.imread('train/A/train_{}.png'.format(i+1))/255
	        img_2 = cv2.imread('train/B/train_{}.png'.format(i+1))/255
	        label = cv2.imread('train/label/train_{}.png'.format(i+1))
	        #print(i)
	        y_label = label[:,:,0:2]
	        y_label[:,:,0][np.where(y_label[:,:,0] == 255)] = 1
	        y_label[:,:,1][np.where(y_label[:,:,1] == 0)] = 1
	        y_label[:,:,1][np.where(y_label[:,:,1] == 255)] = 0

	        xtrain1 += [img_1]
	        xtrain2 += [img_2]
	        labels += [y_label]

	    x_train = np.zeros(shape=(BATCH_SIZE,2,img_sz[0],img_sz[1],N_BANDS),dtype=np.float32)
	    train1 = np.array(xtrain1)
	    train2 = np.array(xtrain2)
	    label = np.array(labels)
	    x_train[:,0,:,:,:] = train1[:,:,:,:]
	    x_train[:,1,:,:,:] = train2[:,:,:,:]
	    yield (x_train,label)
        
def generate_val(N_BANDS, BATCH_SIZE, VAL_DATA, img_sz):
	i = 0
	while 1:
	    xtrain1 = []
	    xtrain2 = []
	    labels = []
	    for b in range(BATCH_SIZE):
	        if i == VAL_DATA:
	            i = 0
	        i += 1

	        y_label = np.zeros(shape=(img_sz[0],img_sz[1],2),dtype=np.float32)
	        img_1 = cv2.imread('val/A/val_{}.png'.format(i+1))/255
	        img_2 = cv2.imread('val/B/val_{}.png'.format(i+1))/255
	        label = cv2.imread('val/label/val_{}.png'.format(i+1))
	        #print(i)
	        y_label = label[:,:,0:2]
	        y_label[:,:,0][np.where(y_label[:,:,0] == 255)] = 1
	        y_label[:,:,1][np.where(y_label[:,:,1] == 0)] = 1
	        y_label[:,:,1][np.where(y_label[:,:,1] == 255)] = 0

	        xtrain1 += [img_1]
	        xtrain2 += [img_2]
	        labels += [y_label]

	    x_train = np.zeros(shape=(BATCH_SIZE,2,img_sz[0],img_sz[1],N_BANDS),dtype=np.float32)
	    train1 = np.array(xtrain1)
	    train2 = np.array(xtrain2)
	    label = np.array(labels)
	    x_train[:,0,:,:,:] = train1[:,:,:,:]
	    x_train[:,1,:,:,:] = train2[:,:,:,:]
	    yield (x_train,label)
        
def get_model():
	return unet(input_size = (2,1024,1024,3))

def main():
    N_BANDS = 3
    BATCH_SIZE = 1
    N_DATA = 440
    VAL_DATA = 60
    img_sz = [1024, 1024]
    EPOCHS = 100000
    model = get_model()
    print('Start Train Net!')
    early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.01,patience=500,mode='min',baseline=20, verbose=1)
    run_model = model.fit_generator(generate(N_BANDS, BATCH_SIZE, N_DATA, img_sz),
                                    steps_per_epoch=int(N_DATA/BATCH_SIZE),
                                    validation_data = generate_val(N_BANDS, BATCH_SIZE, VAL_DATA, img_sz),
                                    validation_steps=int(VAL_DATA/BATCH_SIZE),
                                    shuffle=True, epochs=EPOCHS,
                                    callbacks=[early_stopping]) 
    model.save_weights("model")
    
if __name__ == '__main__':
	main()