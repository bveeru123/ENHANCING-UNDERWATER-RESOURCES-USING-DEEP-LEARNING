import os
import glob
import random
import tensorflow as tf
import cv2
class DataReader:

    

    def readimage(self,img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img,channels=3)
        img = tf.image.resize(img,(400,400))
        img = img / 255.0
        #print(img.shape)
        return img

   
    def generateTrainTestImages(self,hr_trainImages,lr_testImages):
        trainRaw = tf.data.Dataset.from_tensor_slices([img[0] for img in hr_trainImages]).map(lambda name: self.readimage(name))
        trainNormal = tf.data.Dataset.from_tensor_slices([img[1] for img in hr_trainImages]).map(lambda name: self.readimage(name))
        trainData = tf.data.Dataset.zip((trainRaw,trainNormal)).shuffle(100).repeat().batch(8)
        testRaw = tf.data.Dataset.from_tensor_slices([img[0] for img in lr_testImages]).map(lambda name: self.readimage(name))
        testNormal = tf.data.Dataset.from_tensor_slices([img[1] for img in lr_testImages]).map(lambda name: self.readimage(name))
        testData = tf.data.Dataset.zip((testRaw,testNormal)).shuffle(100).repeat().batch(8)
        itr = tf.data.Iterator.from_structure(trainData.output_types,trainData.output_shapes)
        training = itr.make_initializer(trainData)
        testing = itr.make_initializer(testData)
        return training, testing, itr

    def readImages(self,train_images):
        training_path = glob.glob(train_images + "/*.png")
        random.shuffle(training_path)
        trainImages = []
        for name in training_path:
            trainImages.append([name,name])
        return trainImages              
        
        
    
    
