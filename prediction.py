import tensorflow as tf
from model import load_model()
from mirror_pad import post_pad ,remove_pad

def predict(img ,path):
    # Load model and weight
    model = load_model()
    model.load_weights(path)
    
    #Load image and reszie
    img = post_pad(cv2.imread('-Image-path-') ,61)
    same_pic = cv2.imread('-Image-path-')
    pic = same_pic.copy()

    # Patch the images and reshape the image
    patch = patchify(img ,(256 ,256 ,3) ,step =256)
    shape = patch.shape
    stack_img = patch.reshape((shape[0]*shape[1] ,256 ,256,3)) /255

    # predict and reshape the image
    re_patch = model.predict(stack_img).reshape((20, 20, 1, 256, 256, 1))
    unpatch = unpatchify(re_patch ,(5120 ,5120 ,1))
    unpatch = remove_pad(unpatch ,61)


    # Alter the values in predicted
    unpatch[np.where(unpatch < .5)] =0
    unpatch[np.where(unpatch > .5)] =1

    same_pic[np.where(np.sum(unpatch -1 ,axis =-1)==0)] = [91 ,37 ,232] # Building
    #pic1[np.where(np.sum(unpatch -0 ,axis =-1)==0)] = [252 ,221 ,97] # Background
    output =  cv2.addWeighted(same_pic, 0.6,pic, 0.5, 0.0) # for opacity
    return output
