import LoadBatches
from VGGModelUnet import VGGUnet
from VGGModelUnet import set_keras_backend
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from keras_radam import RAdam
import numpy as np
import tensorflow as tf
import os
import cv2
import glob
#import theano.tensor as T


test_images_path = "./test/"
test_segs_path = "./testpredictions/"
test_batch_size = 1

train_images_path = "./train/"
train_segs_path = "./trainmask/"
train_batch_size = 6
n_classes = 2  #7
#n_classes = 4

#input_height = 360
#input_width = 480
input_height = 224
input_width = 224
save_weights_path = "./weights/"
epochs = 600
load_weights = ""
load_weights = "weights/VGGUnet.weights.best.hdf5"
checkpoint_filepath = "weights/VGGUnet.weights.best.hdf5"

val_images_path = "./val/"
val_segs_path = "./valmask/"
val_batch_size = 6


def dice_coef(y_true, y_pred):
    smooth = 1e-8
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection) / (
            K.sum(K.square(y_true_f), -1) + K.sum(K.square(y_pred_f), -1) + smooth)

    #return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_multilabel(y_true, y_pred):
    dice = [0, 0, 0, 0]
    for index in range(n_classes):
        dice -= dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
    return dice


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T



gamma = 2.0
alpha = 0.25

def focal_loss(y_true, y_pred):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow((1 - y_pred), gamma)
    loss = weight * cross_entropy
    loss = K.sum(loss, axis=1)
    return loss

def categorical_focal_loss(y_true, y_pred):

    focal = [0, 0, 0, 0]
    for index in range(n_classes):
        focal -= (focal_loss(y_true[:, :, :, index], y_pred[:, :, :, index]))

    return focal

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes




def test(m, output_width, output_height):
    K.set_learning_phase(0)    

    m.compile(RAdam(lr=1e-4),
              loss=dice_loss, 
              metrics=[dice_coef])


    images = glob.glob(test_images_path + "*.jpg") + glob.glob(test_images_path + "*.png") + glob.glob(test_images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(test_segs_path + "*.jpg") + glob.glob(test_segs_path + "*.png") + glob.glob(test_segs_path + "*.jpeg")
    segmentations.sort()

    for i in range(len(images)):
        im_path=images[i]
        seg_path=segmentations[i]
        data=LoadBatches.getImageArr(im_path, input_width, input_height)
        data=np.expand_dims(data, axis=0) 
        predictions=m.predict(data)
        seg = predictions[0]
        #seg=np.reshape(seg,(output_width, output_height, n_classes))
        threshold=0.5
        img=np.zeros((output_width, output_height, 3),int)
        if (n_classes==2):
            #img[:,:,0]=(seg[:,:,1] >= threshold).astype(int)*255
            #Bladder
            img[:,:,0]=(seg[:,:,1] >= threshold).astype(int)*1
            img[:,:,1]=(seg[:,:,1] >= threshold).astype(int)*1
            img[:,:,2]=(seg[:,:,1] >= threshold).astype(int)*1
        elif (n_classes==4):
            #img[:,:,0]=seg[:,:,1]*255
            #img[:,:,1]=seg[:,:,2]*255
            #img[:,:,2]=seg[:,:,3]*255
            img[:,:,0]=(seg[:,:,1] >= threshold).astype(int)*255
            img[:,:,1]=(seg[:,:,2] >= threshold).astype(int)*255
            img[:,:,2]=(seg[:,:,3] >= threshold).astype(int)*255
        else:
            for c in range(n_classes):
                img[:,:,0]=(seg[:,:,c] >= threshold).astype(int)*c
                img[:,:,1]=(seg[:,:,c] >= threshold).astype(int)*c
                img[:,:,2]=(seg[:,:,c] >= threshold).astype(int)*c
            

        filename = seg_path  
        cv2.imwrite(filename, img) 

    K.clear_session()


def train(m, output_width, output_height):
    #m.compile(RAdam(lr=1e-3),
    #          loss=dice_coef_multilabel, 
    #          metrics=[tversky_loss, dice_coef, dice_coef_multilabel, focal_loss, categorical_focal_loss])
    m.compile(RAdam(lr=1e-4),
              loss=dice_loss, 
              metrics=[tversky_loss, dice_coef, dice_coef_multilabel, focal_loss])

#     train_gen = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, 
#                                                        train_batch_size, n_classes,
#                                                        input_height, input_width, 
#                                                        output_height, output_width)
    train_gen = LoadBatches.augmentedDataGenerator(train_images_path, train_segs_path, 
                                                   train_batch_size, n_classes,
                                                   input_height, input_width, 
                                                   output_height, output_width)
    
    val_gen = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height,
                                                input_width, output_height, output_width)

    print()

    model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True)

    model_tensorboard = TensorBoard(log_dir='./Tensorboard/logs', histogram_freq=0, batch_size=train_batch_size, 
                                        write_graph=True, write_grads=False, write_images=False, 
                                        embeddings_freq=0, embeddings_layer_names=None, 
                                        embeddings_metadata=None, embeddings_data=None, 
                                        update_freq='batch')

    gbytes = get_model_memory_usage(train_batch_size,m)
    print('Total estimated memory GBtytes: %f' % gbytes)

    m.fit_generator(train_gen, steps_per_epoch=int(78 / train_batch_size), validation_data=val_gen, callbacks=[model_checkpoint,model_tensorboard],
                    validation_steps=int(20 / val_batch_size), epochs=epochs)
    m.save_weights(save_weights_path + "finalweights.hdf5")
    m.save(save_weights_path + ".model.finalweights.hdf5")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    set_keras_backend("tensorflow")
    m, output_width, output_height = VGGUnet(n_classes)

    # m.compile(loss=dice_coef_multilabel,
    #           optimizer='adadelta',
    #           metrics=['accuracy'])

    # m.compile(loss='categorical_crossentropy',
    #           optimizer='adadelta',
    #           metrics=['accuracy'])

    if len(load_weights) > 0:
        m.load_weights(load_weights)

    print("Model output shape", m.output_shape)

    print(m.summary(line_length=150))

    #train(m, output_width, output_height)
    test(m, output_width, output_height)

