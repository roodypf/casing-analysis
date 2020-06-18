import cv2
import numpy as np


class analysis(object):
    def __init__(self):
        self.cap = cv2.VideoCapture('C:/Users/tomasz.supernak/Desktop/Nowy folder/Vid2.mp4')
        self.vidcap()
        #self.comparison()

    def vidcap(self):
        while(self.cap.isOpened()):
            # Capture frame-by-frame, edge detection
            _,self.frame = self.cap.read()
            frame = cv2.resize(self.frame, (1024, 800))
            frame = frame[300:550, 550:700]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernel = (7,7)
            blur = cv2.GaussianBlur(frame, kernel, 0)
            edges = cv2.Canny(blur, 80, 200)
            closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            #Identifying edges as points(x and y coordinates) and packing into a list
            indices = np.where(closing != [0])
            coordinates = list(zip(indices[1], indices[0]))
            num = len(coordinates)

            #Separating coordinates into top and bottom edge
            bot_cor = coordinates[:int(num/2)]
            top_cor = coordinates[-int(num/2):]

            #Converting into arrays, sorting
            a, b = np.array(top_cor), np.array(bot_cor)
            a, b = a[a[:,0].argsort()], b[b[:,0].argsort()]
        

    
            #Approximation with a 3rd degree polynomial
            min_a_x, max_a_x = np.min(a[:,0]), np.max(a[:,0])
            new_a_x = np.linspace(min_a_x, max_a_x, 1000)
            a_coefs = np.polyfit(a[:,0],a[:,1], 3)
            new_a_y = np.polyval(a_coefs, new_a_x)

            min_b_x, max_b_x = np.min(b[:,0]), np.max(b[:,0])
            new_b_x = np.linspace(min_b_x, max_b_x, 1000)
            b_coefs = np.polyfit(b[:,0],b[:,1], 3)
            new_b_y = np.polyval(b_coefs, new_b_x)

            #Defining the center line
            midx = [np.average([new_a_x[i], new_b_x[i]], axis = 0) for i in range(1000)]
            midy = [np.average([new_a_y[i], new_b_y[i]], axis = 0) for i in range(1000)]


            #Identifying the coordinates of the centerline, packing into a list
            coords = list(zip(midx, midy))
            points = list(np.int_(coords))

            #Drawing a center line as a series of points (circles)
            for point in points:
                cv2.circle(frame, tuple(point), 1, (255,255,255), -1)

            for point in points:
                cv2.circle(closing, tuple(point), 1, (255,255,255), -1)  


            #Dividing closing by 255 to get 0's and 1's, performing
            #an accumulate addition for each column. 
            a = np.add.accumulate(closing/255,0)
            #Clipping values: anything greater than 2 becomes 2
            a = np.clip(a, 0, 2)
            #Performing a modulo, to get areas alternating with 0 or 1; then multiplying by 255
            a = a%2 * 255
            #Converting to uint8
            mask1 = cv2.convertScaleAbs(a)

            #Flipping the array to get a second mask
            a = np.add.accumulate(np.flip(closing,0)/255,0)
            a = np.clip(a, 0, 2)
            a = a%2 * 255
            mask2 = cv2.convertScaleAbs(np.flip(a,0))
            
            #Summing the intensities of pixels in each mask
            sums = [0,0]
            s1 = (sums[0]) = cv2.sumElems(cv2.bitwise_and(frame,mask1))
            s2 = (sums[1]) = cv2.sumElems(cv2.bitwise_and(frame,mask2))


            cv2.imshow('masked1',cv2.bitwise_and(frame,mask1))
            cv2.imshow('masked2',cv2.bitwise_and(frame,mask2))
            cv2.imshow('mask1', mask1)
            cv2.imshow('mask2', mask2)

            # Display the resulting frame
            cv2.imshow('frame',frame)
            cv2.imshow('closing',closing)

    

            if cv2.waitKey(5) & 0xFF == ord('q'):

                self.cap.release()
                cv2.destroyAllWindows()
            
                
    def comparison():
        s1, s2 = vidcap()
        diff = (s1[0]-s2[0])/(s1[0]+s2[0])
        print(diff, end="\r")
                
if __name__ == '__main__':
    analysis = analysis()
