from tensorflow import keras as ks
import tensorflow as tf
def load_model():
    # Model Intilization

    In = ks.layers.Input((256 ,256 ,3))
    #regulazier = ks.layers.Lambda(lambda In : In/255)(In)

    #Block 1
    con1 = ks.layers.Conv2D(filters = 64 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' ,padding= 'SAME',activation = 'relu')(In)
    drop1 = ks.layers.Dropout(rate =0.5)(con1)
    con1 = ks.layers.Conv2D(filters = 64 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' , padding= 'SAME' ,activation = 'relu')(drop1)
    maxpool1 = ks.layers.MaxPool2D(pool_size=(2,2))(con1)

    #Block 2
    con2 = ks.layers.Conv2D(filters = 128 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu')(maxpool1)
    drop2 = ks.layers.Dropout( rate = .5)(con2)
    con2 = ks .layers.Conv2D(filters = 128 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu')(drop2)
    maxpool2 =ks.layers.MaxPool2D(pool_size=(2,2))(con2)

    # Block 3
    con3 = ks.layers.Conv2D(filters = 256 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu')(maxpool2)
    drop3 = ks.layers.Dropout(rate = .5 ,name = 'drop3')(con3)
    con3 = ks.layers.Conv2D(filters = 256 ,kernel_size = (3,3) , kernel_initializer = 'HeNormal' ,padding= 'SAME' ,activation = 'relu')(drop3)
    maxpool3 =ks.layers.MaxPool2D(pool_size=(2,2))(con3) 

    # Block 4
    con4 = ks.layers.Conv2D(filters = 512 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu')(maxpool3)
    drop4 = ks.layers.Dropout(rate = .5 , name = 'drop4')(con4)
    con4 = ks.layers.Conv2D(filters = 512 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu')(drop4)
    maxpool4 = ks.layers.MaxPool2D(pool_size=(2,2))(con4)

    # Block 5
    con5 = ks.layers.Conv2D(filters = 1024 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu')(maxpool4)
    drop5 = ks.layers.Dropout(rate = .5)(con5)
    con5 = ks.layers.Conv2D(filters = 1024 ,kernel_size = (3,3) ,kernel_initializer = 'HeNormal' , padding= 'SAME',activation = 'relu')(drop5)
    upsamp1 = ks.layers.Conv2DTranspose(filters= 512 ,kernel_size=(2,2) ,strides = (2,2),padding = 'SAME')(con5)

    # UBlock 1
    concat1 = ks.layers.Concatenate(axis = -1)([upsamp1 ,con4])
    cont1 = ks.layers.Conv2D(filters= 512 ,kernel_size=(3,3) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu')(concat1)
    dropt1 = ks.layers.Dropout(rate = .5)(cont1)
    cont1 = ks.layers.Conv2D(filters= 512 ,kernel_size=(3,3) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu')(dropt1)
    upsamp2 = ks.layers.Conv2DTranspose(filters= 256 ,kernel_size=(2,2),strides = (2,2) ,padding = 'SAME')(cont1)

    #UBlock 2
    concat2 = ks.layers.Concatenate(axis = -1)([upsamp2 , con3])
    cont2 = ks.layers.Conv2D(filters = 256 , kernel_size=(3,3) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu')(concat2)
    dropt2 = ks.layers.Dropout(rate = .5)(cont2)
    cont2 = ks.layers.Conv2D(filters = 256 ,kernel_size=(3,3) ,kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu')(dropt2)
    upsmap3 = ks.layers.Conv2DTranspose(filters = 128 ,kernel_size=(2,2),strides = (2,2) ,padding = 'SAME')(cont2)

    #UBlock 3
    concat3 =ks.layers.Concatenate(axis = -1)([upsmap3 , con2])
    cont3 = ks.layers.Conv2D(filters = 128 ,kernel_size=(3,3),kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu')(concat3)
    dropt3 = ks.layers.Dropout(rate =.5)(cont3)
    cont3 = ks.layers.Conv2D(filters = 128 ,kernel_size=(3,3),kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu')(dropt3)
    upsamp4 = ks.layers.Conv2DTranspose(filters = 64 ,kernel_size=(2,2),strides = (2,2) ,padding = 'SAME')(cont3)

    #UBlock 4
    concat4 = ks.layers.Concatenate(axis = -1)([upsamp4 , con1])
    cont4 = ks.layers.Conv2D(filters = 64 ,kernel_size=(3,3),kernel_initializer = 'HeNormal' ,padding = 'SAME',activation = 'relu')(concat4)
    dropt4 = ks.layers.Dropout(rate =.5)(cont4)
    cont4 = ks.layers.Conv2D(filters = 64 ,kernel_size=(3,3),kernel_initializer = 'HeNormal' ,padding = 'SAME' ,activation = 'relu')(dropt4)

    #Output
    Output = ks.layers.Conv2D(filters = 1 ,kernel_size=(1,1) ,kernel_initializer = 'HeNormal' , padding = 'SAME',activation = 'sigmoid')(cont4)

    model = ks.Model(inputs = In ,outputs = Output)
    return model