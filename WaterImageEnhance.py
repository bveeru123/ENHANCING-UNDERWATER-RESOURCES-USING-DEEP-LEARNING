from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from PIL import Image
import matplotlib.pyplot as plt
from DataReader import DataReader
import cv2
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
import os
from keras.models import model_from_json
from utils.data_utils import getPaths, read_and_resize, preprocess, deprocess

main = tkinter.Tk() # blank page 
main.title("Underwater Image Enhancement with Multi-Scale Residual Attention Network")
main.geometry("1200x1200")


global multi_patch_model, saver, RGB, MAX, filename, itr

def getMultpatchModel(RGB):
    cnn1 = Conv2D(3,1,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(RGB)#layer 1
    cnn2 = Conv2D(3,3,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(cnn1)#layer 2
    fa_block1 = tf.concat([cnn1,cnn2],axis=-1) #concatenate layer1 and layer2 to from residual network
    cnn3 = Conv2D(3,5,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fa_block1)
    fa_block2 = tf.concat([cnn2,cnn3],axis=-1)#concatenate layer2 and layer3 to from residual network
    cnn4 = Conv2D(3,7,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fa_block2)
    CA = tf.concat([cnn1,cnn2,cnn3,cnn4],axis=-1)
    cnn5 = Conv2D(3,3,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(CA)
    MAX = cnn5 #max layer
    multipatch = ReLU(max_value=1.0)(tf.math.multiply(MAX,RGB) - MAX + 1.0) #replace pixels intensity
    return multipatch

def uploadDataset():
    global filename, itr
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir = ".")
    dr = DataReader()  #class to read training images
    tf.reset_default_graph() #reset tensorflow graph
    trainImages = dr.readImages('Dataset/reference-890')
    testImages = dr.readImages('Dataset/raw-890')
    trainData, testData, itr = dr.generateTrainTestImages(trainImages,testImages)
    text.insert(END,"Training & Testing Images Loaded\n\n")
    text.insert(END,"Total Training Images : "+str(len(trainImages))+"\n")
    text.insert(END,"Total Testing Images  : "+str(len(testImages))+"\n")    

def PSNR(original, compressed):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)  
    mse_value = np.mean((original - compressed) ** 2) 
    if(mse_value == 0):
        return 100
    max_pixel = 255.0
    psnr_value = 100 - (20 * log10(max_pixel / sqrt(mse_value))) 
    return psnr_value / 2

def imageSSIM(normal, embed):
    normal = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY)
    embed = cv2.cvtColor(embed, cv2.COLOR_BGR2GRAY) 
    ssim_value = ssim(normal, embed, data_range = embed.max() - embed.min())
    return ssim_value


def loadModel():
    global multi_patch_model,saver, RGB, MAX
    text.delete('1.0', END)
    next_element = itr.get_next()
    RGB = tf.placeholder(shape=(None,400, 400,3),dtype=tf.float32)
    MAX = tf.placeholder(shape=(None,400, 400,3),dtype=tf.float32)
    multi_patch_model = getMultpatchModel(RGB) #loading and generating multi patch model
    trainingLoss = tf.reduce_mean(tf.square(multi_patch_model-MAX)) #optimizations
    optimizerRate = tf.train.AdamOptimizer(1e-4)
    trainVariables = tf.trainable_variables()
    gradient = optimizerRate.compute_gradients(trainingLoss,trainVariables)
    clippedGradients = [(tf.clip_by_norm(gradients,0.1),var1) for gradients,var1 in gradient]
    optimize = optimizerRate.apply_gradients(gradient)
    saver = tf.train.Saver()
    pathlabel.config(text='Multi-patch Model loaded')
    with open('models/gen_p/model_15320_.json', "r") as json_file:
        loaded_model_json = json_file.read()
    json_file.close()    
    multi_patch_model = model_from_json(loaded_model_json)
    multi_patch_model.load_weights('models/gen_p/model_15320_.h5')
    orig = cv2.imread("Dataset/reference-890/100_img_.png")
    height, width, channels = orig.shape
    orig = cv2.resize(orig,(256, 256),interpolation = cv2.INTER_CUBIC)
    inp_img = read_and_resize("Dataset/raw-890/100_img_.png", (256, 256))
    im = preprocess(inp_img)
    im = np.expand_dims(im, axis=0) 
    gen = multi_patch_model.predict(im)
    enhance_Image = deprocess(gen)[0]
    propose_psnr = PSNR(orig, enhance_Image)
    propose_ssim = imageSSIM(orig, enhance_Image)
    text.insert(END,"Propose Multi-Patch Model PSNR : "+str(propose_psnr)+"\n")
    text.insert(END,"Propose Multi-Patch Model SSIM : "+str(propose_ssim))   
 

#function to allow user to upload images directory
def uploadImage():
    text.delete('1.0', END)
    global multi_patch_model,saver, RGB, MAX
    filename = askopenfilename(initialdir = "testImages")
    pathlabel.config(text=filename)
    inp_img = read_and_resize(filename, (256, 256))
    im = preprocess(inp_img)
    im = np.expand_dims(im, axis=0) 
    enhance_Image = multi_patch_model.predict(im)
    enhance_Image = deprocess(enhance_Image)[0]
    enhance_Image = cv2.resize(enhance_Image,(256, 256),interpolation = cv2.INTER_CUBIC)
    enhance_Image = cv2.cvtColor(enhance_Image, cv2.COLOR_BGR2RGB)


    orig = cv2.imread(filename)
    orig = cv2.resize(orig,(256, 256),interpolation = cv2.INTER_CUBIC)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    propose_psnr = PSNR(orig, enhance_Image)
    propose_ssim = imageSSIM(orig, enhance_Image)
    text.insert(END,"Propose Multi-Patch Model PSNR : "+str(propose_psnr)+"\n")
    text.insert(END,"Propose Multi-Patch Model SSIM : "+str(propose_ssim)) 
    figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(8,8))
    axis[0].set_title("Original Image")
    axis[1].set_title("Enhanced Image")
    axis[0].imshow(orig)
    axis[1].imshow(enhance_Image)
    figure.tight_layout()
    plt.show()
    
def close():
    main.destroy()

font = ('times', 20, 'bold')
title = Label(main, text='Underwater Image Enhancement with Multi-Scale Residual Attention Network')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=5,y=5)

font1 = ('times', 14, 'bold')

upload = Button(main, text="Upload UIEB Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

upload = Button(main, text="Generate & Load Multi-Patch Model", command=loadModel)
upload.place(x=50,y=150)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=380,y=150)

dcpButton = Button(main, text="Upload & Enhance Image", command=uploadImage)
dcpButton.place(x=50,y=200)
dcpButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=250)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=350)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
