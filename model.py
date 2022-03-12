from tensorflow import keras as ks
import tensorflow as tf
from typing import *
def load_model(input_shape:list , total_classes:int):
    r'''Arg:
        Model input shape and Number of Classes .
        example : ( (256 ,256 , 3) ,3)

        Return : 
        UNet  Model 
    '''
    # Model Intilization

    In = Input(input_shape)

    #Block 1
    con1  = Conv2D(filters = 1 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' ,padding= 'SAME',activation = 'relu' )(In)
    con1 =Conv2D(filters = 64 ,kernel_size = (1,1) ,kernel_initializer = 'HeNormal' ,padding= 'SAME',activation = 'relu' )(con1)
    drop1 = Dropout(rate =0.5 )(con1)
    con1 = Conv2D(filters = 1 ,kernel_size = (3,3)  ,kernel_initializer = 'HeNormal' , padding= 'SAME' ,activation = 'relu' )(drop1)
    con1 = Conv2D(filters = 64 ,kernel_size = (1,1)  ,kernel_initializer = 'HeNormal' , padding= 'SAME' ,activation = 'relu' )(con1)
    maxpool1 = MaxPool2D(pool_size=(2,2) )(con1)

    #Block 2
    con2 = Conv2D(filters = 1 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu' )(maxpool1)
    con2 = Conv2D(filters = 128 ,kernel_size = (1,1) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu' )(con2)
    drop2 = Dropout( rate = .5 )(con2)
    con2 = Conv2D(filters = 1 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(drop2)
    con2 = Conv2D(filters = 128 ,kernel_size = (1,1) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(con2)
    maxpool2 = MaxPool2D(pool_size=(2,2) )(con2)

    # Block 3
    con3 = Conv2D(filters = 1 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu' )(maxpool2)
    con3 = Conv2D(filters = 256 ,kernel_size = (1,1) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu' )(con3)
    drop3 = Dropout(rate = .5  )(con3)
    con3 = Conv2D(filters = 1 ,kernel_size = (3,3) , kernel_initializer = 'HeNormal' ,padding= 'SAME' ,activation = 'relu' )(drop3)
    con3 = Conv2D(filters = 256 ,kernel_size = (1,1) , kernel_initializer = 'HeNormal' ,padding= 'SAME' ,activation = 'relu' )(con3)
    maxpool3 =MaxPool2D(pool_size=(2,2) )(con3) 

    # Block 4
    con4 = Conv2D(filters = 1 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu' )(maxpool3)
    con4 = Conv2D(filters = 512 ,kernel_size = (1,1) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu' )(con4)
    drop4 = Dropout(rate = .5  )(con4)
    con4 = Conv2D(filters = 1 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu' )(drop4)
    con4 = Conv2D(filters = 512 ,kernel_size = (1,1) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu' )(con4)
    maxpool4 = MaxPool2D(pool_size=(2,2) )(con4)

    # Block 5
    con5 = Conv2D(filters = 1 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal',name = 'block5' , padding= 'SAME',activation = 'relu' )(maxpool4)
    con5 = Conv2D(filters = 1024 ,kernel_size = (1,1) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu' )(con5)
    drop5 = Dropout(rate = .5 )(con5)
    con5 = Conv2D(filters = 1 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu' )(drop5)
    con5 = Conv2D(filters = 1024 ,kernel_size = (1,1) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu' )(con5)
    upsamp1 = Conv2DTranspose(filters= 512 ,kernel_size=(2,2) ,strides = (2,2),padding = 'SAME' )(con5)

    # UBlock 1
    concat1 = Concatenate(axis = -1)([upsamp1 ,con4] )
    cont1 = Conv2D(filters= 1 ,kernel_size=(3,3) ,kernel_initializer = 'HeNormal' ,name = 'ublock1',padding = 'SAME',activation = 'relu' )(concat1)
    cont1 = Conv2D(filters= 512 ,kernel_size=(1,1) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(cont1)
    dropt1 = Dropout(rate = .5 )(cont1)
    cont1 = Conv2D(filters= 1 ,kernel_size=(3,3) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(dropt1)
    cont1 = Conv2D(filters= 512 ,kernel_size=(1,1) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(cont1)
    upsamp2 = Conv2DTranspose(filters= 256 ,kernel_size=(2,2),strides = (2,2) ,padding = 'SAME' )(cont1)
    #UBlock 2
    concat2 = Concatenate(axis = -1)([upsamp2 , con3])
    cont2 = Conv2D(filters = 1 , kernel_size=(3,3) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(concat2)
    cont2 = Conv2D(filters = 256 , kernel_size=(1,1) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(cont2)
    dropt2 = Dropout(rate = .5 )(cont2)
    cont2 = Conv2D(filters = 1 ,kernel_size=(3,3) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(dropt2)
    cont2 = Conv2D(filters = 256 ,kernel_size=(1,1) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(cont2)
    upsmap3 = Conv2DTranspose(filters = 128 ,kernel_size=(2,2),strides = (2,2) ,padding = 'SAME' )(cont2)

    #UBlock 3
    concat3 =Concatenate(axis = -1)([upsmap3 , con2])
    cont3 = Conv2D(filters = 1 ,kernel_size=(3,3),kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(concat3)
    cont3 = Conv2D(filters = 128 ,kernel_size=(3,3),kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(cont3)
    dropt3 = Dropout(rate =.5 )(cont3)
    cont3 = Conv2D(filters = 1 ,kernel_size=(3,3),kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(dropt3)
    cont3 = Conv2D(filters = 128 ,kernel_size=(1,1),kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(cont3)
    upsamp4 = Conv2DTranspose(filters = 64 ,kernel_size=(2,2),strides = (2,2) ,padding = 'SAME' )(cont3)

    #UBlock 4
    concat4 = Concatenate(axis = -1)([upsamp4 , con1])
    cont4 = Conv2D(filters = 1 ,kernel_size=(3,3),kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(concat4)
    cont4 = Conv2D(filters = 64 ,kernel_size=(1,1),kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu' )(cont4)
    dropt4 = Dropout(rate =.5 )(cont4)
    cont4 = Conv2D(filters = 1 ,kernel_size=(3,3),kernel_initializer = 'HeNormal' ,padding = 'SAME' ,activation = 'relu' )(dropt4)
    cont4 = Conv2D(filters = 64 ,kernel_size=(1,1),kernel_initializer = 'HeNormal' ,padding = 'SAME' ,activation = 'relu' )(cont4)
    upsamp4 = Conv2DTranspose(filters = 64 ,kernel_size=(2,2),strides = (2,2) ,padding = 'SAME' )(cont3)


    #Output
    Output = Conv2D(filters = total_classes ,kernel_size=(1,1) ,kernel_initializer = 'HeNormal' , padding = 'SAME',activation = 'sigmoid')(upsamp4)

    model = Model(inputs = In ,outputs = Output)

    return model
