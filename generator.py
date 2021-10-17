for mirror_pad import pad

def datagen():
    lis=[]
    num =0
    while True:
        for i in glob.glob('*.tif'):lis.append(i) # change the format of the photo here I'm using tif format
        random.shuffle(lis)

        for pic in lis:
            x_train =[] ;y_train = []

            x_train = patchify(cv2.imread(pic) ,(254 ,254 ,3) ,step =254) .reshape(361 ,254 ,254,3) /255 

            y_train = patchify(cv2.imread(f'/content/change the path /ytrain/{pic}') ,(254 ,254 ,3) ,step =254) .reshape(361 ,254 ,254,3) /255  # change the path 

        for _ in range(2):y_train = np.delete(y_train ,1 ,axis =-1) 
      
        temp =0
        for j in range(31,370 ,30):

            padding = pad(x_train[temp:j] ,y_train[temp:j],2)
            yield padding[0] ,padding[1]
            temp=j
        
            num+=1
            print(end = '=')
        print(f"------{num} Batches Completed -----")
        


def valgen():
    lis=[]
    while True:
        for i in glob.glob('/content/change the path/val_x/*.tif'):lis.append(i) # change the path
        random.shuffle(lis)
    
        for pic in lis:
            x_val =[] ;y_val = []

            x_val = (patchify(cv2.imread(pic) ,(256 ,256 ,3) ,step =256)).reshape(361,256,256,3) /255

            y_val = (patchify(cv2.imread(pic[0:31]+'val_y'+pic[36:]) ,(512 ,512 ,3) ,step =512)).reshape(361,512,512,3) /255

            for _ in range(2):y_val = np.delete(y_val ,1 ,axis =-1)
            temp =0

            for j in range(31,361 ,30):
                yield x_val[temp:j] ,y_val[temp:j]
                temp=j