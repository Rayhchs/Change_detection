import numpy as np
import keras
import cv2
from keras.models import *
from model import unet
import os 
from keras import backend as K

os = os.getcwd()
PATH_testA = os + '/test/A/'
PATH_testB = os + '/test/B/'
PATH_label = os + '/test/label/'
PATH_output = os + '/test/predict/'
N_TEST = 120
img_sz = 1024

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true)+K.sum(y_pred)-intersection
    iou = K.mean((intersection + smooth) / (union + smooth))
    return iou

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    dice = K.mean((2. * intersection + smooth)/(union + smooth))
    return dice

def get_model():
	return unet(input_size = (2,1024,1024,3))

def main():
    iou_total = []
    F1_total = []
    for i in range(0,N_TEST):
        img_1 = cv2.imread(PATH_testA + 'test_{}.png'.format(i+1))
        img_2 = cv2.imread(PATH_testB + 'test_{}.png'.format(i+1))
        mask = cv2.imread(PATH_label + 'test_{}.png'.format(i+1))
        test = np.zeros(shape=(1,2,img_sz,img_sz,3),dtype=np.float32)
        test[0,0,:,:,:] = img_1/255
        test[0,1,:,:,:] = img_2/255
        base_model = get_model()
        base_model.load_weights('model')
        predict = base_model.predict(test)
        prediction = predict[0,:,:,0]
        for j in range(prediction.shape[0]):
            for k in range(prediction.shape[1]):
                if prediction[j,k] > 0.5:
                    prediction[j,k] = 1
                else:
                    prediction[j,k] = 0  
        predicts = prediction.copy()
        predicts[:,:][np.where(predicts[:,:] == 1)] = 255 
        cv2.imwrite(PATH_output + '{}.png'.format(i+1),np.uint8(predicts))
        
        mask = np.float32(mask[:,:,0])
        mask[:,:][np.where(mask[:,:] == 255)] = 1
        iou = iou_coef(mask,prediction,smooth=1)
        F1 = dice_coef(mask,prediction,smooth=1)
        iou_total.append(iou)
        F1_total.append(F1)
        
    iou_total = np.array(iou_total)
    F1_total = np.array(F1_total)
    final_output = np.zeros(shape=(F1_total.shape[0],2),dtype=np.float32)
    final_output[:,0] = iou_total
    final_output[:,1] = F1_total
    np.savetxt("evaluation.csv", final_output, delimiter=",")

if __name__ == '__main__':
	main()


