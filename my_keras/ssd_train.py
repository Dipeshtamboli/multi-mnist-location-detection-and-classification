import pdb
import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
# from scipy.misc import imread
from skimage.transform import resize as imresize
import skimage
# from scipy.misc import imresize
from imageio import imread
# from imageio import imresize
import tensorflow as tf

from model.ssd300MobileNetV2Lite import SSD as SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility

# %matplotlib inline
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
# pdb.set_trace()
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))



# some constants
NUM_CLASSES = 10+1
# input_shape = (224, 224, 3)
input_shape = (300, 300, 3)

priors = pickle.load(open('priorFiles/prior_boxes_ssd300MobileNetV2.pkl', 'rb'))
# pdb.set_trace()
priors = priors[:1692]
bbox_util = BBoxUtility(NUM_CLASSES, priors)

# gt = pickle.load(open('voc_2007.pkl', 'rb'))
# keys = sorted(gt.keys())
# num_train = int(round(0.8 * len(keys)))
# train_keys = keys[:num_train]
# val_keys = keys[num_train:]
# num_val = len(val_keys)

def load_gt_bbox(csv_path, base_folder_path):
# gt_file = open("/home/dipesh/Desktop/rohan/mnist/data.csv", "r")
    gt_file = open(csv_path, "r")
    imagename = []
    num_objects = []
    class_numbers = []
    x_y_centres = []
    rotation = []
    size = []
    for sample in gt_file:
        name = sample.split(',')[0]
        imagename.append(name)
        
        objects_present = int(sample.split(',')[1])
        num_objects.append(objects_present)
        
        class_numbers_present = sample.split(',')[2:2+objects_present]
        class_numbers.append(class_numbers_present)

        x_y_centres_present = sample.split(',')[2+objects_present:2+2*objects_present]
        size_present = sample.split(',')[2+2*objects_present:2+3*objects_present]
        rotation_present = sample.split(',')[2+3*objects_present:2+4*objects_present]

        x_y_centres.append(x_y_centres_present)
        size.append(size_present)
        rotation.append(rotation_present)

    data_len = len(imagename)
    # t_num = 100
    # data_len = t_num
    # zip_out = zip(imagename[:t_num], num_objects[:t_num], class_numbers[:t_num], x_y_centres[:t_num], size[:t_num], rotation[:t_num])
    zip_out = zip(imagename, num_objects, class_numbers, x_y_centres, size, rotation)
    return zip_out, data_len

# zip_gt_bbox, len_data = load_gt_bbox("/home/dipesh/Desktop/rohan/mnist/data.csv","/home/dipesh/Desktop/rohan/mnist/multi_imgs")
# for imagename, num_objects, class_numbers, x_y_centres, size, rotation in zip_gt_bbox:
#     # pdb.set_trace()
#     path_prefix = "/home/dipesh/Desktop/rohan/mnist/multi_imgs/"
#     img_path = path_prefix + imagename
#     img = imread(img_path).astype('float32')

#     y = np.zeros((num_objects, 4 + 10))
#     for obj_sample in range(num_objects):
#         y[obj_sample][0]= int(x_y_centres[obj_sample].split('-')[0]) - int(size[obj_sample])//2
#         y[obj_sample][1]= int(x_y_centres[obj_sample].split('-')[1]) - int(size[obj_sample])//2
#         y[obj_sample][2]= y[obj_sample][0] + int(size[obj_sample])
#         y[obj_sample][3]= y[obj_sample][1] + int(size[obj_sample])
#         # pdb.set_trace()
#         # y[obj_sample][4:] = np.eye(10)[int(class_numbers[obj_sample])]
#         y[obj_sample][4+int(class_numbers[obj_sample])] = 1.0
#     y[:,:4] = np.floor(y[:,:4] *300/224)
#     pdb.set_trace()
class Generator(object):
    def __init__(self, csv_path, folder_path_prefix,batch_size, image_size):
        self.csv_path = csv_path
        self.folder_path_prefix = folder_path_prefix
        self.batch_size = batch_size
        self.zip_gt_bbox, self.train_batches = load_gt_bbox(csv_path,folder_path_prefix)
        self.bbox_util = bbox_util
        self.image_size = image_size
    def generate(self, train=True):
        while True:        
            inputs = []
            targets = []
            for imagename, num_objects, class_numbers, x_y_centres, size, rotation in self.zip_gt_bbox:
                img_path = self.folder_path_prefix + imagename
                img = imread(img_path).astype('float32')
                img = imresize(img, self.image_size).astype('float32')
                # print(f"input img shape: {img.shape}")
                # img = np.concatenate([img for i in range(3)])
                img = skimage.color.gray2rgb(img)
                # print(f"input img shape: {img.shape}")
                # pdb.set_trace()
                y = np.zeros((num_objects, 4 + 10))
                for obj_sample in range(num_objects):
                    y[obj_sample][0]= int(x_y_centres[obj_sample].split('-')[0]) - int(size[obj_sample])//2
                    y[obj_sample][1]= int(x_y_centres[obj_sample].split('-')[1]) - int(size[obj_sample])//2
                    y[obj_sample][2]= y[obj_sample][0] + int(size[obj_sample])
                    y[obj_sample][3]= y[obj_sample][1] + int(size[obj_sample])
                    # y[obj_sample][4:] = np.eye(10)[int(class_numbers[obj_sample])]
                    y[obj_sample][4+int(class_numbers[obj_sample])] = 1.0
                y[:,:4] = np.floor(y[:,:4] *300/224)
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)                
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets                

csv_path = "../mnist/multi_mnist/train_data.csv"
folder_path_prefix = "../mnist/multi_mnist/train_multi_imgs/"
val_csv_path = "../mnist/multi_mnist/val_data.csv"
val_folder_prefix = "../mnist/multi_mnist/val_multi_imgs/"
# csv_path = "/home/dipesh/Desktop/rohan/mnist/small_data.csv"
# folder_path_prefix = "/home/dipesh/Desktop/rohan/mnist/small_multi_imgs/"
gen = Generator(csv_path, folder_path_prefix, 1, (input_shape[0], input_shape[1]))


model = SSD300(input_shape, num_classes=NUM_CLASSES)
# model.load_weights('VGG16SSD300_weights_voc_2007.hdf5', by_name=True)

freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']#,
#           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

for L in model.layers:
    if L.name in freeze:
        L.trainable = False

def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

callbacks = [keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule)]
# callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
#                                              verbose=1,
#                                              save_weights_only=True),
#              keras.callbacks.LearningRateScheduler(schedule)]
base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
# optim = keras.optimizers.RMSprop(lr=base_lr)
# optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss,metrics=['acc'])

# nb_epoch = 30
nb_epoch = 3
history = model.fit_generator(gen.generate(True), 10000,
                              nb_epoch, verbose=1,
                              callbacks=callbacks,
                              validation_data=gen.generate(False),
                              nb_val_samples=500,
                              nb_worker=1)

val_gt_bbox, len_data = load_gt_bbox(val_csv_path,val_folder_prefix)

inputs = []
images = []
count = 0
num_img_count =10
for imagename, num_objects, class_numbers, x_y_centres, size, rotation in val_gt_bbox:
    # pdb.set_trace()
    # img = imread(img_path).astype('float32')

    # img_path = folder_path_prefix + sorted(val_keys)[0]
    path_prefix = val_folder_prefix
    img_path = path_prefix + imagename
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_path))
    inputs.append(img.copy())
    count += 1
    if count>= num_img_count:
        break
inputs = preprocess_input(np.array(inputs))

preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)

conf_thres = 0.3
save_count = 0
for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    # top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    # pdb.set_trace()
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thres]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
#         label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        # print(label)
        # color = colors[label]
        color = colors[0]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    
    # plt.show()
    plt.savefig("{}.jpg".format(save_count))
    save_count += 1
    plt.savefig("default.jpg")
    # pdb.set_trace()

# def det_and_save(conf_thres = 0.17):
