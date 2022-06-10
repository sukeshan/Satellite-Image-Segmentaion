
# Calculate area and number of buildings using Satellite Image Segmentation  
 ![image](https://user-images.githubusercontent.com/48553042/163168539-1e74324e-c9c0-43d0-a039-715f51e7f119.png)
### Annotated dataset       :  https://project.inria.fr/aerialimagelabeling/files/
### Pretrained weight files :  https://drive.google.com/file/d/10R8BKgnfCAmylta9-iWSi_eiXFIo1JKW/view?usp=sharing

# Description 
  Used U-net architecture to segment the satellite images . Dataset used is from Inria (open-source platform) . Since satellite images have high resolution, it cannot be fit into the memory hence divided each images into smallpatches which are than stitched after prediction. Applied mirror padding to preserve smooth connection between patches when stitched together, synthetically generated data to overcome class imbalance. Due to low computation resources used depthwise separable convolution which reduced model size from 31 to 2.8 million parameterswithout impacting the accuracy . Achieved 89% validation accuracy.

## Install Requirements
    pip instal -r requirements.txt

## Web_Page 
      streamlit run app.py
      
## Model Prediction:
    from prediction import predict
    img = 'image file path'
    weight = 'model pretrained weigth file path'
    result = predict(img ,weight)

##  Train a model with your own dataset
### Load Modules
    from model import load_model
    from generator import data_genertor
    from loss import focal_loss
    from metrics import IoU ,recall,precission ,f1score
 
 ## Model fit
    model = load_model()
    x_train = data_generator(img_dir:list ,input_size :tuple )
    y_train = data_generator(img_dir:list ,input_size :tuple )
    model.compile(optimizer="adam" ,loss=focl_loss ,metrics = [IoU ,recall,precission ,f1score])
    model.fit_generator( x_trin ,y_train ,batch_size =16 ,epochs = 50)
    
 
