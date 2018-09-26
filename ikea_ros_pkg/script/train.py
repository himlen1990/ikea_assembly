from read_data import dataReader
import numpy as np
import tensorflow as tf
from model import regression_model
import skimage.io as io
import cv2

def train():
    sess = tf.Session()
    model = regression_model(sess)
    dr = dataReader('./dataset')
    
    iteration = 15000
    saver = tf.train.Saver()
    test_image_batch, test_label_batch = dr.get_test_data(10)
    #test_image_batch = test_image_batch - 144
    for i in range(iteration):
        train_image_batch, train_label_batch = dr.sample_batch(5)
        #train_image_batch = train_image_batch-144
        model.learn(train_image_batch, train_label_batch)
        if i % 20 == 0:
            print "iteration--- ",i
            print "train loss"
            model.eval(train_image_batch, train_label_batch)
            print "test loss"
            model.eval(test_image_batch, test_label_batch)
    saver.save(sess, './params')
    #test_image = image_batch[-1,:,:,:]
    #result = model.predict(test_image)
    #print result
    #a = test_image.astype(np.uint8)
    #io.imshow(a)
    #io.show()



if __name__=='__main__':
    train()

    
