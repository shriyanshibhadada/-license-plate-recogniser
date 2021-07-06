import cv2
import DetectPlates


def detect(cap):
    img_array = []
    while cap.isOpened():
        ret , img = cap.read()
        #img = cv2.resize(img, ( int(img.shape[1]/2), int(img.shape[0]/2)))
        if ret == True:
            #new = img.copy()
            #ans = DetectPlates.locateLP(img)
            ans = DetectPlates.final_img_and_number(img)
            cv2.imshow('img', ans)


            height, width, layers = ans.shape
            size = (width, height)
            img_array.append(ans)

            #cv2.imshow('img', new)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        if cap.isOpened() == False:
            break

    out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__' :
    cap = cv2.VideoCapture('video2.mp4')
    detect(cap)
    #cap.release()
    #cv2.destroyAllWindows()