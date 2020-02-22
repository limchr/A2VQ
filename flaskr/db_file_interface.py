import numpy as np
import pandas as pd

import pickle as pkl
import dill


from flaskr.settings import *

from common.helper import setup_clean_directory
from common.data_handler.create_arbitrary_image_ds import feature_extraction_of_arbitrary_image_ds
from common.helper import save_csv,read_csv

import PIL.Image as Image

def write_image(img,file):
    if np.max(img) <= 1: # is float array
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
    else:
        pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img.save(file)


def init():
    '''init classifier, delete old files'''

    setup_clean_directory(DUMP_PATH)

    feats, image_list, img_tuple = feature_extraction_of_arbitrary_image_ds(IMAGE_PATH)


    num_samples = len(image_list)
    labels = np.zeros(num_samples, dtype=str)
    label_file = np.vstack((image_list, labels)).T
    save_csv(label_file, LABEL_FILE)


    with open(FEATURES_FILE,'wb') as f:
        pkl.dump(feats, f)


    with open(IMAGES_FILE,'wb') as f:
        pkl.dump(img_tuple, f)



    from machine_learning_models.glvq import glvq
    cls = glvq()
    with open(CLASSIFIER_FILE, 'wb') as f:
        dill.dump(cls,f)



def update_labels(indices,label):
    labels = read_csv(LABEL_FILE)
    labels[indices,1] = label
    save_csv(labels, LABEL_FILE)



def load_features():
    with open(FEATURES_FILE, 'rb') as f:
        feats = pkl.load(f)
    return feats

def update_embedding(x_embedding_normalized):
    with open(EMBEDDING_FILE, 'wb') as f:
        pkl.dump(x_embedding_normalized, f)

def load_embedding():
    with open(EMBEDDING_FILE, 'rb') as f:
        emb = pkl.load(f)
    return emb


def load_images():
    '''create thumbnails not already there of all images for showing over HTTP'''
    with open(IMAGES_FILE, 'rb') as f:
        images_file = pkl.load(f)
    return images_file


def get_labels():
    '''return all labels'''
    return read_csv(LABEL_FILE)[:, 1]


def _load_classifier():
    with open(CLASSIFIER_FILE, 'rb') as f:
        cls = dill.load(f)
    return cls

def _dump_classifier(cls):
    with open(CLASSIFIER_FILE, 'wb') as f:
        cls = dill.dump(cls,f)

def classifier_partial_fit(x,y):
    cls = _load_classifier()
    cls.partial_fit(x,y)
    _dump_classifier(cls)

def classifier_predict(x):
    cls = _load_classifier()
    return cls.predict(x)

def classifier_predict_proba(x):
    cls = _load_classifier()
    return cls.predict_proba(x)

def get_indices():
    '''return all image filenames'''
    return read_csv(LABEL_FILE)[:, 0]


##############################################
### extra functions only for this interface###
##############################################





def create_class_wise_db():
    from functions import get_unlabeled_mask
    from common.images import write_image
    setup_clean_directory(CLASS_WISE_DB_PATH)
    labels = np.array(get_labels())
    imgs = np.array(load_images())
    filenames = np.array(get_indices())
    unlabeled_mask = get_unlabeled_mask()

    labels[unlabeled_mask] = '_UNLABELED_'

    unique_labels = np.unique(labels)

    for label in unique_labels:
        setup_clean_directory(os.path.join(CLASS_WISE_DB_PATH, label))
        inds = np.where(labels == label)[0]
        for filename,img in zip(filenames[inds], imgs[inds]):
            write_image(img, os.path.join(CLASS_WISE_DB_PATH, label,filename))


from common.helper import get_files_of_type



def label_db():
    db = pkl.load(open(DB_FILE,'rb'))

    def get_immediate_subdirectories(a_dir):
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

    classes = get_immediate_subdirectories(CLASS_WISE_DB_PATH)

    del (classes[classes.index("_UNLABELED_")])

    for cls in classes:
        img_filenames = get_files_of_type(os.path.join(CLASS_WISE_DB_PATH, cls), '.jpg')
        indices = [x[:-4] for x in img_filenames]
        for i in indices:
            try:
                db.loc[db.index == i, 'label'] = cls
            except ValueError:
                print('DANGER: INDEX '+str(i)+' IS NOT IN DATABASE')

    db.to_pickle(os.path.join(DUMP_PATH,'labeled.db'))

def export_db():
    setup_clean_directory(EXPORT_PATH)
    setup_clean_directory(os.path.join(EXPORT_PATH,'images'))

    db = pkl.load(file(os.path.join(DUMP_PATH,'labeled.db'),'rb'))
    #drop samples where there was no label given
    db = db.drop(db.index[db.label.isnull()])
    #export features
    import h5py
    h5f = h5py.File(os.path.join(EXPORT_PATH, 'features.hdf'), 'w')
    h5f.create_dataset('index', data=db.index.to_numpy(dtype='string'))
    h5f.create_dataset('feats', data=np.stack(db.feats.to_numpy(),axis=0))
    h5f.close()
    # db.feats.to_hdf(os.path.join(EXPORT_PATH, 'features.hdf'), 'features', mode='w', format='table')

    imgs = get_files_of_type(IMAGE_PATH, '.jpg')



    #drop different lines from property file that are not needed for export
    properties_file = db.drop(columns=['feats', 'img', 'embedding_x', 'embedding_y', 'label_pred', 'confidence_pred', 'instance_id', 'index'])


    properties_file.to_csv(os.path.join(EXPORT_PATH,'properties.csv'))
    properties_file.to_pickle(os.path.join(EXPORT_PATH, 'properties.pkl'))


    #skip the images that not belong to a samples in property file
    remove_imgs = []

    for i in range(len(imgs)):
        if not imgs[i][:-4] in properties_file.index:
            remove_imgs.append(imgs[i][:-4])

    from shutil import copyfile

    for img in properties_file.index: # also to ensure that all images are there
        if not img in remove_imgs:
            copyfile(os.path.join(IMAGE_PATH, img+'.jpg'), os.path.join(EXPORT_PATH,'images',img+'.jpg'))





if __name__ == '__main__':
    print('main')
    # create_class_wise_db()
    # label_db()
    export_db()
    #db = pkl.load(open(os.path.join(DUMP_PATH,'labeled.db'),'rb'))
