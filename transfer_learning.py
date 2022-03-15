from model import load_model
from tensorflow.keras.optimizers import Adam
from typing import *

def pre_train(base_model :object ,input_shape :tuple ,xtrain :object ):
    r''''Arg:
        base_model = loaded model from model.py
        
        input_shape = Input shape
        
        xtrain = loaded Image
        
        Return:
        model with learned weights.
        
        example : transfer_learning(base_model = load_model(input_shape ,classes) ,input_shape ,xtrain )
                
    '''
    #Intialize model for Image Reconstruction:
    
    trans_model = load_model(input_shape = input_shape ,classes = 3)
    trans_model.compile(optimizer=Adam(learning_rate=0.01) ,loss = 'mean_squared_error' ,metrics=['accuracy'])
    
    print('\t\t\t\t\t','-'*8,"TRANSFER_LEARNING_STARTED",'-'*8)
    trans_model.fit(xtrain,xtrain,epochs = 2 )
    print('\n\t\t\t\t\t','-'*8,"TRANSFER_LEARNING_COMPLETED",'-'*8)
    
    #Transfer the encoder weight to base_model:
    i = 0
    for weight in trans_model.trainable_weights[:30]:
        base_model.trainable_weights[i].assign(weight)
        i+=1
    return base_model  
