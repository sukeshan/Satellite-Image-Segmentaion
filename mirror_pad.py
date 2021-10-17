import numpy as np

def pad(x_train ,y_train ,pad_size):
    xt = []
    yt = []
    for x in range(len(x_train)):
        temp0 = np.concatenate((x_train[x][::-1][-pad_size:-1] ,x_train[x]) ,0)
        temp1 = np.concatenate((temp0 ,x_train[x][::-1][1:pad_size]),0)
        temp2 = np.concatenate((temp1[:,::-1][: ,-pad_size:-1] ,temp1) ,1)
        temp3 =np.concatenate((temp2 ,temp1[:,::-1][: ,1:pad_size]) ,1)
        xt.append(temp3)
    for y in range(len(y_train)):
        temp0 = np.concatenate((y_train[y][::-1][-pad_size:-1] ,y_train[y]) ,0)
        temp1 = np.concatenate((temp0 ,y_train[y][::-1][1:pad_size]),0)
        temp2 = np.concatenate((temp1[:,::-1][: ,-pad_size:-1] ,temp1) ,1)
        temp3 =np.concatenate((temp2 ,temp1[:,::-1][: ,1:pad_size]) ,1)
        yt.append(temp3)
    xt = np.array(xt)
    yt = np.array(yt)
    return xt ,yt

def post_pad(x_train  ,pad_size):
    xt = []
    temp0 = np.concatenate((x_train[::-1][-pad_size:-1] ,x_train) ,0)
    temp1 = np.concatenate((temp0 ,x_train[::-1][1:pad_size]),0)
    temp2 = np.concatenate((temp1[:,::-1][: ,-pad_size:-1] ,temp1) ,1)
    temp3 =np.concatenate((temp2 ,temp1[:,::-1][: ,1:pad_size]) ,1)
    xt.append(temp3)
    xt = np.array(xt)

    return xt.reshape(xt.shape[1:])

def remove_pad(arr ,padded_size):
    padded_size -=1 
    arr = arr[padded_size:]
    arr = arr[:-padded_size]
    arr = arr[:,padded_size:]
    arr = arr[:,:-padded_size]
    return arr
