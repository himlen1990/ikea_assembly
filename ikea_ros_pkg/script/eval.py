
import numpy as np
import tensorflow as tf
from model import regression_model
import skimage.io as io
from skimage.draw import circle

def show_label(image,predict,ground_true):


    print "predict"
    mask = predict[...,0]
    cell_length = image.shape[0]/mask.shape[0] 
    x = predict[...,1]
    y = predict[...,2]
    mask_f = mask.flatten()
    top_n = 1
    n_max = mask_f.argsort()[-top_n:]
    print mask_f[n_max]
    top_n_idx = []
    for i in range(top_n):
        idx = np.unravel_index(n_max[i],mask.shape)
        top_n_idx.append(idx)
    print  top_n_idx
    for i in range(0, len(top_n_idx)):
        circle_x = x[top_n_idx[i][0]][top_n_idx[i][1]] + top_n_idx[i][1]*cell_length
        circle_y = y[top_n_idx[i][0]][top_n_idx[i][1]] + top_n_idx[i][0]*cell_length
        rr,cc = circle(circle_y, circle_x, 3)
        print circle_x,circle_y
        image[rr,cc] = (220,20,20)
                       
    """
    print "ground_true"
    mask = ground_true[...,0]
    x = ground_true[...,1]
    y = ground_true[...,2]
        
    row,col = np.where(mask > 0.0)
    for i in range(row.shape[0]):            
    circle_x = x[row[i]][col[i]] + col[i]*cell_length
    circle_y = y[row[i]][col[i]] + row[i]*cell_length
    print circle_x, circle_y
    rr,cc = circle(circle_y, circle_x, 3)
    image[rr,cc] = (20,220,20)
    """
    io.imshow(image)
    io.show()


def eval():

    sess = tf.Session()
    model = regression_model(sess)
    saver = tf.train.Saver()
    saver.restore(sess, "./params")
    #test_images, test_labels = dr.get_test_data(20)
    img_file = ("./frame0402.jpg")
    #img_file = ("./dataset/frame0100.jpg")
    test_image = io.imread(img_file)
    #label_file = img_file[:-4]+".npy"
    #test_label = np.load(label_file)
    test_label = []
    test_idx = 1
    cell_size = 8
    boundary = cell_size * cell_size
    result = model.predict(test_image)
    mask = np.reshape(result[:,:boundary],[cell_size,cell_size])
    x = np.reshape(result[:,boundary:boundary*2],[cell_size,cell_size])
    y = np.reshape(result[:,boundary*2:],[cell_size,cell_size])
    restruct_label = np.stack((mask,x,y),axis=-1)
    
    #print test_label
    #print restruct_label
    show_label(test_image,restruct_label,test_label)
    #io.imshow(test_images[0])
    #io.show()


if __name__=='__main__':
    #train()
    eval()
