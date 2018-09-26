import tensorflow as tf
import numpy as np



class regression_model:
    def __init__(self,sess):
        self.cell_size = 8
        self.sess = sess
        self.label_ph = tf.placeholder(tf.float32, shape=(None,self.cell_size,self.cell_size,3))
        self.input_ph = tf.placeholder(tf.float32, shape=(None,448,448,3))
      
        self.boundary = self.cell_size * self.cell_size

        self.num_output = self.cell_size * self.cell_size * 3
        out = self.input_ph
        out = tf.layers.conv2d(out, 4, 5, padding='same')
        out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
        out = tf.layers.batch_normalization(out, training=True)
        out = tf.nn.relu(out)

        for _ in range(3):
            out = tf.layers.conv2d(out, 16, 3, padding='same')
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)


        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        #out = tf.layers.dense(out,64, activation=tf.nn.relu)
        out = tf.layers.dense(out,512, activation=tf.nn.relu)
        self.output = tf.layers.dense(out,self.num_output, name="final")

        self.object_mask = self.label_ph[...,0]
        self.x = self.label_ph[...,1]
        self.y = self.label_ph[...,2]

        #test---------------------
        mask_indices = tf.where(self.object_mask > 0.0)
        self.x_hat = tf.gather_nd(self.x , mask_indices)
        self.y_hat = tf.gather_nd(self.y , mask_indices)

        #------------------------
        output_object_mask = tf.reshape(self.output[:,:self.boundary],[-1,self.cell_size,self.cell_size])
        output_x = tf.reshape(self.output[:,self.boundary:self.boundary*2],[-1,self.cell_size,self.cell_size])
        output_y = tf.reshape(self.output[:,self.boundary*2:],[-1,self.cell_size,self.cell_size])
        print output_object_mask.shape
        print self.object_mask.shape
        #test ----------------------
        output_x_hat = tf.gather_nd(output_x , mask_indices)
        output_y_hat = tf.gather_nd(output_y , mask_indices)
        #-----------------------------

        mask_loss = tf.reduce_sum(tf.square(output_object_mask - self.object_mask))
        #x_loss = tf.reduce_sum(tf.square(output_x - self.x))
        x_loss = tf.reduce_sum(tf.square(output_x_hat - self.x_hat))
        #y_loss = tf.reduce_sum(tf.square(output_y - self.y))
        y_loss = tf.reduce_sum(tf.square(output_y_hat - self.y_hat))
        self.loss = mask_loss + x_loss + y_loss
        self.train_op = tf.train.AdamOptimizer(epsilon=0.1).minimize(self.loss)
        sess.run(tf.global_variables_initializer())


    def predict(self, image):
        return self.sess.run(self.output, feed_dict={self.input_ph: [image]})


    def predict_batch(self,images_batch):
        return self.sess.run(self.output, feed_dict=image_batch)


    def learn(self, images_batch, label_batch):
        #print(self.sess.run(self.loss, {self.input_ph: images_batch, self.label_ph: label_batch}))
        self.sess.run(self.train_op, {self.input_ph: images_batch, self.label_ph: label_batch})

    def eval(self, images_batch, label_batch):
        print(self.sess.run(self.loss, {self.input_ph: images_batch, self.label_ph: label_batch}))


    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, 'params')

'''
sess = tf.Session()
model = regression_model(sess)


test = np.array([[[1,48.65,106.0696,-0.424],[0.,0.,0.,0.]],[[1,6.65,135.779,15.06],[0.,0.,0.,0.]]])
print test
print (sess.run(model.object_mask, {model.label_ph: [test]}))
print (sess.run(model.grasp, {model.label_ph: [test]}))
'''
