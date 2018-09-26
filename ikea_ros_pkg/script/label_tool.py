import cv2
import numpy as np
import os
import glob
import numpy 

ix,iy = -1,-1

dataset_dir = './dataset'
finish_click = False



def draw_circle(event,x,y,flags,param):
    global ix,iy, finish_click

    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
        finish_click = True

    elif event == cv2.EVENT_MOUSEMOVE:
        pass
    elif event == cv2.EVENT_LBUTTONUP:
        pass


filenames = []
for name in  glob.glob(os.path.join(dataset_dir, '*.jpg')):
    filenames.append(name)

filenames = sorted(filenames, key=lambda name: int(name[-8:-4]))


img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)


go_on = True
label = np.zeros((4,2))
count = 0
for image in filenames:
    if go_on == False:
        break
    print image
    img = cv2.imread(image)
    while(go_on):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if finish_click:
            print "x: ", ix, "y: ", iy
            cv2.circle(img,(ix,iy),5,(0,0,255),-1)
            label[count] = np.array([ix,iy])
            count = count + 1
            if count >4:
                print "please renotate"
                count = 0
                img = cv2.imread(image)
            finish_click = False
        #if k == ord('o'):
        #    finish_click = True
        if k == ord('c'): #cancle
            count = 0
            img = cv2.imread(image)
        elif k == ord('n'):
            print label
            label_name = image[:-4] + '.npy'
            np.save(label_name, label)
            count = 0
            label = np.zeros((4,2))
            break
        elif k == 27:
            go_on = False
            break

cv2.destroyAllWindows()
