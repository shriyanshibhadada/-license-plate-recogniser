#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import main


# In[2]:


net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg", r"yolov3_custom_2000.weights")


# In[3]:


def show(img):
    print(img.shape)
    plt.imshow(img)
    plt.show()


# In[4]:


classes = ['licence']


# In[5]:


#img = cv2.imread('image4.jpg')


# In[7]:


def locateLP(img):
    hight,width,_ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)
    net.setInput(blob)
    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)


    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5:
                #print(confidence)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
    #print(indexes)
    ans = []
    #detectedimg = img.copy()
    #font = cv2.FONT_HERSHEY_PLAIN
    if  len(indexes)>0:
        for i in indexes.flatten():
            # x,y,w,h = boxes[i]
            ans.append(boxes[i])
            # label = str(classes[class_ids[i]])
            # confidence = str(round(confidences[i],2))
            # color = (255,255,255)
            # cv2.rectangle(detectedimg,(x,y),(x+w,y+h),color,10)
            # cv2.putText(detectedimg,label + " " + confidence, (x,y+400),font,2,color,2)
            # show(detectedimg)
    
    ans = np.array(ans)
    return ans


def final_img_and_number(img):
    ans = locateLP(img)

    PlateNumber = []
    new = img.copy()
    for i in ans:
        x, y, w, h = i
        #x, y, w, h = ans[0][0], ans[0][1], ans[0][2], ans[0][3]
        cv2.rectangle(new, (x, y), (x + w, y + h), (255, 255, 255), 5)
        #number, segments = main.PlateRecognition(img[y:y + h, x:x + w])
        #FinalImage.append(new)
        #PlateNumber.append(number)

    return new #, PlateNumber


if __name__ == '__main__':
    img = cv2.imread('ps2/ps2/test_multipleCar/p3.png')
    ans, PlateNumber = final_img_and_number(img)
    show(ans)
    for i in PlateNumber:
        print(i)