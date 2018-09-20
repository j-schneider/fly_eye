
from __future__ import print_function
from keras.utils import np_utils
from keras.optimizers import adam, rmsprop, SGD
from keras.datasets import cifar10
from custom_image import DoubleDirImageDataGenerator, DAImageDataGenerator
from custom_image import ImageDataGenerator  # Preserve Aspect when zooming
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
import os
from keras import backend as K
import numpy as np
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import skimage.transform
import tensorflow as tf
from keras.utils import multi_gpu_model
from models import Zeiler_Fergus, ResnetBuilder, DAResnetBuilder, fly_eye
from models import CustomStopper, MultiGPU_Checkpoint_Callback
# from skimage.transform import resize as imresizeyy

parser = argparse.ArgumentParser(description='Keras Fly-Eye models')
parser.add_argument('--descriptor', default='date', type=str,
                    help='Characteristic name for the experiment - to be used \
                    for log files, checkpoints, etc')
parser.add_argument('--model', default='fly-eye', type=str,
                    help='Model name fly-eye, Zeiler_Fergus, ResNet18, \
                    or DA-ResNet18)')
parser.add_argument('--data', default='cifar10', type=str,
                    help='Data source, cifar10 or flypics')
parser.add_argument('--week', default=1, type=int,
                    help='Which week to test (if flies)')
parser.add_argument('--epochs', default=100, type=int,
                    help='Potential number of epochs to run')
parser.add_argument('--min_epochs', default=10, type=int, metavar='MIN_EPOCH',
                    help='Minimum number of epochs to run')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='Mini-batch size (default: 64)')
parser.add_argument('--opt', default='SGD', type=str,
                    metavar='OPT', help='Optimizer (default: SDG)')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR',
                    help='Initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum (only used if --opt SGD)')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--pixels', default=29, type=int,
                    help='How many pixels to use (n x n)')
parser.add_argument('--resize', default=None, type=int,
                    help='Resize the images loaded (default=pixels)')
parser.add_argument('--evaluate', default=None, type=str,
                    help='Filename to evaluate (weights)')
parser.add_argument('--zoom_range', default=0., type=float, metavar='ZOOM',
                    help='Float determining zooming of image')
parser.add_argument('--use_multi', default=0, type=int,
                    help='Use multi GPUs for training')
parser.add_argument('--load_weights', default=None, type=str,
                    help='File to load weights from prior to training')
parser.add_argument('--pretrain_outputs', default=0, type=int,
                    metavar='PREOUT',
                    help='The number of outputs of the weight file, when used\
                    with "load_weights", can adapt wieghts to current number\
                    of outputs.')
parser.add_argument('--patience', default=0, type=int,
                    help='Epochs of patience before early stopping')
parser.add_argument('--pic_dir', default='/mnt/data/jschne02/Data/Fly_Pics/',
                    type=str, help='Directory of the pictures')


global args
args = parser.parse_args()
week = args.week
if args.resize is None:
    args.resize = args.pixels
if args.descriptor == 'date':
    import datetime
    args.descriptor = datetime.date.isoformat(datetime.date.today())
if args.data == 'flies':
    model_name = '%s_%s_%s-pixels_%s-resize_%s-week%s' % \
        (args.descriptor, args.model,
            args.pixels, args.resize, args.data, args.week)
else:
    model_name = '%s_%s_%s-pixels_%s-resize_%s' % \
        (args.descriptor, args.model, args.pixels, args.resize, args.data)

if args.zoom_range != 0.:
    model_name = model_name + '_%s-zoom' % args.zoom_range

log_dir = './log/' + model_name

print('Building %s with a resolution of %s pixels' % (args.model, args.pixels))
if args.pixels is not args.resize and args.data == 'flies':
    print('Forcing images through a resize of %sx%s pixels' %
          (args.resize, args.resize))
print('Training on: %s' % args.data)
if args.evaluate is None:
    print('Logging tensorboard data to:%s' % log_dir)

batch_size = args.batch_size
epochs = args.epochs
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models_files')

img_size = args.pixels

if args.data == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_predictions = 10
    # Randomly split off 5000 of training for validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1)
    # Convert class vectors to binary class matrices.
    y_train = np_utils.to_categorical(y_train, num_predictions)
    y_val = np_utils.to_categorical(y_val, num_predictions)
    y_test = np_utils.to_categorical(y_test, num_predictions)
    if args.resize != 0:
        x_train = np.asarray([skimage.transform.resize(
            image, (args.resize, args.resize, 3), anti_aliasing=True,
            mode='constant') for image in x_train])
        x_val = np.asarray([skimage.transform.resize(
            image, (args.resize, args.resize, 3), anti_aliasing=True,
            mode='constant') for image in x_val])
        x_test = np.asarray([skimage.transform.resize(
            image, (args.resize, args.resize, 3), anti_aliasing=True,
            mode='constant') for image in x_test])
    # subtract mean and normalize
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image
    x_train /= 128.
    x_val /= 128.
    x_test /= 128.
else:
    num_predictions = 20

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

if args.use_multi > 0:
    with tf.device('/cpu:0'):
        if args.model == 'fly-eye':
            cpumodel = fly_eye(img_size, num_predictions, args.resize,
                               args.pretrain_outputs, args.load_weights)
        elif args.model == 'Zeiler_Fergus':
            cpumodel = Fergus_Zeiler(img_size, num_predictions,
                                     args.resize, args.pretrain_outputs,
                                     args.load_weights)
        elif args.model == 'ResNet18':
            cpumodel = ResnetBuilder.build_resnet_18(img_size, num_predictions,
                                                     args.resize)
        elif args.model == 'DA-ResNet18':
            cpumodel = DAResnetBuilder.build_resnet_18(img_size,
                                                       num_predictions,
                                                       args.resize)
    if args.load_weights is not None and args.pretrain_outputs is 0:
        cpumodel.load_weights(args.load_weights)

else:
    if args.model == 'fly-eye':
        model = fly_eye(img_size, num_predictions, args.resize,
                        args.pretrain_outputs, args.load_weights)
    elif args.model == 'Zeiler_Fergus':
        model = Fergus_Zeiler(img_size, num_predictions,
                              args.resize, args.pretrain_outputs,
                              args.load_weights)
    elif args.model == 'ResNet18':
        model = ResnetBuilder.build_resnet_18(img_size, num_predictions,
                                              args.resize)
    elif args.model == 'DA-ResNet18':
        model = DAResnetBuilder.build_resnet_18(img_size,
                                                num_predictions,
                                                args.resize)
    if args.load_weights is not None and args.pretrain_outputs is 0:
        model.load_weights(args.load_weights)

if args.opt == 'SGD':
    # use this for DA
    opt = SGD(lr=args.lr, decay=args.wd, momentum=args.momentum, nesterov=True)
elif args.opt == 'adam':
    # need this for fly-eye
    opt = adam(lr=args.lr, decay=args.wd)
elif args.opt == 'rmsprop':
    opt = rmsprop(lr=args.lr, decay=args.wd)

if args.evaluate is not None:
    if args.use_multi > 0:
        cpumodel.load_weights(args.evaluate)
    else:
        model.load_weights(args.evaluate)

if args.use_multi > 0:
    model = multi_gpu_model(cpumodel, gpus=args.use_multi)
    if args.model == 'DA-ResNet18':
        model.compile(
            loss={'ID': 'categorical_crossentropy', 'Domain':
                  'binary_crossentropy'},
            optimizer=opt,
            metrics={'ID': 'accuracy', 'Domain': 'accuracy'})
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
else:
    if args.model == 'DA-ResNet18':
        model.compile(
            loss={'ID': 'categorical_crossentropy', 'Domain':
                  'binary_crossentropy'},
            optimizer=opt,
            metrics={'ID': 'accuracy', 'Domain': 'accuracy'})
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])


def preprocess_input(x):
    # subtract mean and divide by std of the training set of fly images
    x -= 178.62
    x /= 29.09
    return x


# Setup the data generators
if args.evaluate is None:
    if args.model == 'DA-ResNet18':
        # DAImageDataGenerator is set up for two directories,
        # and returns binary label depending on which
        img_train_datagen = DAImageDataGenerator(
            zoom_range=args.zoom_range,
            horizontal_flip=True,  # randomly flip images
            preprocessing_function=preprocess_input)

        img_val_datagen = DAImageDataGenerator(
            preprocessing_function=preprocess_input)

    elif args.data == 'flies':
        img_train_datagen = DoubleDirImageDataGenerator(
            zoom_range=args.zoom_range,
            horizontal_flip=True,  # randomly flip images
            preprocessing_function=preprocess_input)

        img_val_datagen = DoubleDirImageDataGenerator(
            preprocessing_function=preprocess_input)

    elif args.data == 'cifar10':
        img_train_datagen = ImageDataGenerator(
            zoom_range=args.zoom_range,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)  # randomly flip images

        img_val_datagen = ImageDataGenerator()

if args.data == 'cifar10':
    b_train = int(x_train.shape[0]/batch_size)
    b_val = int(x_val.shape[0]/batch_size)
    b_test = int(x_test.shape[0]/batch_size)

    if args.evaluate is None:
        traingenerator = img_train_datagen.flow(x_train, y_train,
                                                batch_size=batch_size)
        valgenerator = img_val_datagen.flow(x_val, y_val,
                                            batch_size=batch_size,
                                            shuffle=False)
    img_test_datagen = ImageDataGenerator(
        zoom_range=args.zoom_range)
    testgenerator = img_test_datagen.flow(x_test, y_test,
                                          batch_size=batch_size,
                                          shuffle=False)
else:
    b_train = int(489520/batch_size)
    b_val = int(86400/batch_size)
    b_test = int(287980/batch_size)

    if args.evaluate is None:
        traingenerator = img_train_datagen.flow_from_directory(
            (args.pic_dir + 'week%s/Day1/train/' % week,
             args.pic_dir + 'week%s/Day2/train/' % week),
            # save_to_dir='pics/',
            target_size=(args.resize, args.resize), class_mode='categorical',
            batch_size=batch_size)
        valgenerator = img_val_datagen.flow_from_directory(
            (args.pic_dir + 'week%s/Day1/val/' % week,
             args.pic_dir + 'week%s/Day2/val/' % week),
            target_size=(args.resize, args.resize), class_mode='categorical',
            batch_size=batch_size, shuffle=False)

    img_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)
    testgenerator = img_test_datagen.flow_from_directory(
        args.pic_dir + 'week%s/Day3/' % week,
        target_size=(args.resize, args.resize), class_mode='categorical',
        batch_size=batch_size, shuffle=False)

if args.evaluate is None:
    # Callbacks

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
                                   min_lr=0.5e-6)
    early_stopper = CustomStopper(min_delta=0, patience=5,
                                  start_epoch=args.min_epochs)
    tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0,
                             write_graph=False, write_images=True)
    csv_logger = CSVLogger(os.path.join(save_dir, model_name+'.csv'))

    if args.use_multi > 0:
        bestCallBack = MultiGPU_Checkpoint_Callback(
            os.path.join(save_dir, model_name+'-weights-best.hdf5'),
            base_model=cpumodel, monitor='val_loss', verbose=0,
            save_best_only=True, save_weights_only=False, mode='auto',
            period=1)
        model.fit_generator(traingenerator,
                            epochs=epochs,
                            steps_per_epoch=b_train,
                            validation_data=valgenerator,
                            validation_steps=b_val,
                            workers=4, callbacks=[tbCallBack, bestCallBack,
                                                  lr_reducer, early_stopper,
                                                  csv_logger])

        cpumodel.save(model_path+'.h5')
        print('Saved final trained model at %s ' % model_path)
        # load the best weights after training
        cpumodel.load_weights(os.path.join(save_dir, model_name +
                              '-weights-best.hdf5'))
    else:
        bestCallBack = ModelCheckpoint(
            os.path.join(save_dir, model_name+'-weights-best.hdf5'),
            monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', period=1)
        model.fit_generator(traingenerator,
                            epochs=epochs,
                            steps_per_epoch=b_train,
                            validation_data=valgenerator,
                            validation_steps=b_val,
                            workers=4, callbacks=[tbCallBack, bestCallBack,
                                                  lr_reducer, early_stopper,
                                                  csv_logger])

        model.save(model_path+'.h5')
        print('Saved final trained model at %s ' % model_path)
        # load the best weights after training
        model.load_weights(os.path.join(save_dir, model_name +
                           '-weights-best.hdf5'))

text_file = open(os.path.join(save_dir, model_name+'_report.txt'), 'a+')

pred_list = np.array([])
label_list = np.array([])
batches = 0

if args.model == 'DA-ResNet18':
    # Will predict class and domain, only interested in class
    for x_batch, y_batch in testgenerator:
        pred = model.predict(x_batch)
        pred_class = pred[0].argmax(axis=-1)
        pred_list = np.append(pred_list, np.array(pred_class))
        y_class = y_batch.argmax(axis=-1)
        label_list = np.append(label_list, np.array(y_class))
        batches += 1
        if batches >= b_test:
            break
else:
    for x_batch, y_batch in testgenerator:
        pred = model.predict(x_batch)
        pred_class = pred.argmax(axis=-1)
        pred_list = np.append(pred_list, np.array(pred_class))
        y_class = y_batch.argmax(axis=-1)
        label_list = np.append(label_list, y_class)
        batches += 1
        if batches >= b_test:
            break

text_file.write('Scores:\n')
text_file.write(classification_report(label_list, pred_list, digits=4))
text_file.write('Confusion Matrix:\n')

cf = confusion_matrix(label_list, pred_list) / 1.
for x in range(np.shape(cf)[0]):
    cf[x, :] = cf[x, :]/float(np.sum(cf[x, :]))

np.savetxt(text_file, cf, fmt='%3.2f', delimiter=',')

text_file.close()
