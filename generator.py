import numpy as np
import cv2
import random 
from patchify import patchify ,unpatchify
from sklearn.utils import shuffle
from padding import pad

def data_generator(img_dir:list ,input_size :tuple ,batch_size :int ,pad = False)  -> Union[np.ndarray ,np.ndarray]  :
    r''' Arg :
         img_dir :Image Directory Path for both Input image and labeled image in list format.
         
         input_size : Input Size of the Image in Tuple format. 
         
         batch_size : Batch Size.
         
         pad : If pad is True, the mirror padding will happen.
         
         example : datagen(img_dir_path =[Input_image_path ,Output_image_path] ,input_size = (256,256,3) ,batch_size = 30 ,pad =False)
         
         Return :
         Xtrain , ytrain : Batch of Images return in array
         
         
    '''
    # PAtch the Images
    def patch(Image:np.ndarray ,row ,col ):
        x_train = patchify(Image ,(row ,col ,3) ,step = row) .reshape(total_slice ,row ,col,3) /255 
        y_train = patchify((labeled_image) ,(row ,col ,3) ,step =row) .reshape(total_slice ,row ,col,3) /255  # change the pat
    
    
    num =0
    while True:
        # List the Input Images Name
        data_points =[]
        for i in os.listdir(img_dir_path[0]):data_points.append(i) 
        data_points = shuffle(data_points)
        
        for pic in data_points:
            
            x_train =[] ;y_train = []
            if pad is True: pixel =2
            else: pixel = 0
                
            # Set the row and column of the images   
            row = input_size[0]-pixel
            col = input_size[1]-pixel
            
            Image = cv2.imread(pic) 
            labeled_image = cv2.imread(cv2.imread(img_dir_path[1]+f'{pic}') 
            total_slice = int((image.shape[0])/row)**2                     
                                      
            x_train ,y_train = patch(Image ,row ,col )
                                       
            # Change the dimension into One i.e (256,256,3) <- (256,256,1)                            
            for _ in range(2):y_train = np.delete(y_train ,1 ,axis =-1) 
            temp =0

            for j in range(batch_size+1 ,total_slice ,batch_size):

                padding = pad(x_train[temp:j] ,y_train[temp:j],pixel)
                x ,y =shuffle(padding[0] ,padding[1])        
                                       
                yield x,y
                temp=j ;num+=1
                print(end = '=')
            print(f"------{num} Batches Completed -----")

        
