import skimage.io as io
from skimage.draw import circle
from skimage.transform import resize
import numpy as np
import glob
import os
import random
import cv2

class dataReader(object):
    def __init__(self, dataset_dir):
        images = []
        labels = []
        filenames = []
        cell_size = 8
        label_processed = []
        image_size = 448
        cell_length = image_size/cell_size
        
        for name in glob.glob(os.path.join(dataset_dir, '*.jpg')):
            filenames.append(name)

        for filename in filenames:
            img = cv2.imread(filename)    
            label_file = filename[:-4]+'.npy'       
            label = np.load(label_file)
            images.append(img)
            labels.append(label)
        
        
        for label in labels:
            new_label = np.zeros((cell_size,cell_size,3))
            for i in range(label.shape[0]):
                x = label[i,0]
                x_ind = int(x * cell_size/image_size)
                bias_x = x - x_ind *cell_length
                y = label[i,1]
                y_ind = int(y * cell_size/image_size)
                bias_y = y - y_ind *cell_length
                location = [bias_x,bias_y]
                new_label[y_ind,x_ind,0] = 10
                new_label[y_ind,x_ind,1:3] = location
            label_processed.append(new_label)


        self.all_images = np.array(images)
        self.all_labels = np.array(label_processed)
        print self.all_images.shape
        print self.all_labels.shape

        #self.show_label(self.all_images[0],self.all_labels[0])
        train_set, test_set = self.split_dataset()
        self.train_img, self.train_label = zip(*train_set)
        self.test_img, self.test_label = zip(*test_set)
        self.train_img = np.array(self.train_img)
        self.train_label = np.array(self.train_label)
        self.test_img = np.array(self.test_img)
        self.test_label = np.array(self.test_label)

    def split_dataset(self,num_train=280):
        self.dataset = zip(self.all_images, self.all_labels)
        random.shuffle(self.dataset)
        return self.dataset[:num_train], self.dataset[num_train:]


    def get_test_data(self,batch_size = 10):
        idx = np.random.choice(self.test_img.shape[0], batch_size)
        return self.test_img[idx],self.test_label[idx]

    def sample_batch(self, batch_size=10):
        idx = np.random.choice(self.train_img.shape[0], batch_size)
        return self.train_img[idx], self.train_label[idx]


    def show_label(self,image,predict,predict2,predict3):


        print "predict"
        mask = predict[...,0]
        cell_length = image.shape[0]/mask.shape[0] 
        x = predict[...,1]
        y = predict[...,2]
        mask_f = mask.flatten()
        top_n = 3
        n_max = mask_f.argsort()[-top_n:]
        top_n_idx = []
        for i in range(top_n):
            idx = np.unravel_index(n_max[i],mask.shape)
            top_n_idx.append(idx)

        for i in range(0, len(top_n_idx)):
            circle_x = x[top_n_idx[i][0]][top_n_idx[i][1]] + top_n_idx[i][1]*cell_length
            circle_y = y[top_n_idx[i][0]][top_n_idx[i][1]] + top_n_idx[i][0]*cell_length
            #rr,cc = circle(circle_y, circle_x, 3)
            #image[rr,cc] = (220,20,20)
            cv2.circle(image,(int(circle_x),int(circle_y)),3,(0,255,0),-1)


        mask = predict2[...,0]
        cell_length = image.shape[0]/mask.shape[0] 
        x = predict2[...,1]
        y = predict2[...,2]
        mask_f = mask.flatten()
        top_n = 3
        n_max = mask_f.argsort()[-top_n:]
        top_n_idx = []
        for i in range(top_n):
            idx = np.unravel_index(n_max[i],mask.shape)
            top_n_idx.append(idx)

        for i in range(0, len(top_n_idx)):
            circle_x = x[top_n_idx[i][0]][top_n_idx[i][1]] + top_n_idx[i][1]*cell_length
            circle_y = y[top_n_idx[i][0]][top_n_idx[i][1]] + top_n_idx[i][0]*cell_length
            #rr,cc = circle(circle_y, circle_x, 3)
            #image[rr,cc] = (220,20,20)
            cv2.circle(image,(int(circle_x),int(circle_y)),3,(255,0,0),-1)


        mask = predict3[...,0]
        cell_length = image.shape[0]/mask.shape[0] 
        x = predict3[...,1]
        y = predict3[...,2]
        mask_f = mask.flatten()
        top_n = 3
        n_max = mask_f.argsort()[-top_n:]
        top_n_idx = []
        for i in range(top_n):
            idx = np.unravel_index(n_max[i],mask.shape)
            top_n_idx.append(idx)

        for i in range(0, len(top_n_idx)):
            circle_x = x[top_n_idx[i][0]][top_n_idx[i][1]] + top_n_idx[i][1]*cell_length
            circle_y = y[top_n_idx[i][0]][top_n_idx[i][1]] + top_n_idx[i][0]*cell_length
            #rr,cc = circle(circle_y, circle_x, 3)
            #image[rr,cc] = (220,20,20)
            cv2.circle(image,(int(circle_x),int(circle_y)),3,(0,0,255),-1)


        cv2.imshow("result",image)
        cv2.waitKey(0)
        #io.show()
        
                         
'''
dr = dataReader('./dataset')

train_set, test_set = dr.split_dataset()
train_img, train_label = zip(*train_set)
test_img, test_label = zip(*test_set)
print np.array(train_img).shape
print np.array(train_label).shape
print np.array(test_img).shape
print np.array(test_label).shape

train_img = np.array(train_img)
test_img = train_img[0]
train_label = np.array(train_label)
test_label = train_label[0]

x = test_label[1,:,:][np.where(test_label[1,:,:] > 0.0)]
y = test_label[2,:,:][np.where(test_label[2,:,:] > 0.0)]

print test_label[1,:,:]
print np.where(test_label[1,:,:] > 0.0)
print np.where(test_label[2,:,:] > 0.0)
print x[0],y[0]
#io.imshow(test_img)
#io.show()
'''
#read_dataset('./dataset')
