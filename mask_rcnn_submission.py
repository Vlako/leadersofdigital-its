#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

from skimage.io import imshow
from skimage.transform import resize
from skimage.color import rgb2grey
from skimage.morphology import erosion
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

import os
import io
from random import choice
from itertools import product
from base64 import b64encode, b64decode
from collections import Counter

from tqdm import tqdm


# In[2]:


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[3]:


loaded = tf.saved_model.load('mask_rcnn_inception_resnet_v2/saved_model/')
infer = loaded.signatures["serving_default"]


# In[4]:


mapper = [
    'none',
    'staphylococcus_aureus',
    'staphylococcus_epidermidis',
    'ent_cloacae',
    'c_kefir',
    'moraxella_catarrhalis',
    'klebsiella_pneumoniae'
]


# In[5]:


def tta(image, vertical_flip=False, horizontal_flip=False, rotate=False):
    if vertical_flip:
        image = cv2.flip(image, 0)
    if horizontal_flip:
        image = cv2.flip(image, 1)
    if rotate:
        image = np.rot90(image)
    return image

def transform_tta_mask(mask, vertical_flip=False, horizontal_flip=False, rotate=False):
    if rotate:
        mask = np.rot90(mask, -1)
    if horizontal_flip:
        mask = cv2.flip(mask, 1)
    if vertical_flip:
        mask = cv2.flip(mask, 0)
    return mask


# In[6]:


result = []
sample = pd.read_csv('bacteria.csv', dtype={'id': str})
for imagename in tqdm(sample['id']):
    image = cv2.imread(f'test/test/{imagename}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    labels = Counter()

    result_mask = np.zeros(image.shape[:2])
    
    for vert, hor, rot in product([False, True], repeat=3):
        tta_image = tta(image, vert, hor, rot)
        tta_mask = np.zeros(tta_image.shape[:2])
        
        raw_result = infer(tf.convert_to_tensor(np.expand_dims(tta_image, axis=0)))

        for score, label, mask, box in zip(
            raw_result['detection_scores'][0].numpy(), 
            raw_result['detection_classes'][0].numpy(), 
            raw_result['detection_masks'][0].numpy(),
            raw_result['detection_boxes'][0].numpy()):

            if score < 0.35:
                continue

            labels[label] += 1

            y_min, x_min = int(box[0] * tta_image.shape[0]), int(box[1] * tta_image.shape[1])
            y_max, x_max = int(box[2] * tta_image.shape[0]), int(box[3] * tta_image.shape[1])

            mask = resize(mask, (y_max - y_min, x_max - x_min))
            mask = (mask > 0.35).astype(int)

            tta_mask[y_min:y_max, x_min:x_max] = mask
        
        result_mask += transform_tta_mask(tta_mask, vert, hor, rot)
        
    result_mask = np.clip(result_mask, 0, 1)
        
    label = mapper[int(labels.most_common()[0][0])]
    
    mask_bytes = io.BytesIO()

    Image.fromarray((result_mask * 255).astype(np.uint8), mode='P').save(mask_bytes, format='PNG')
    
    result.append({
        'id': imagename.replace('.png', ''),
        'class': label,
        'base64 encoded PNG (mask)': b64encode(mask_bytes.getvalue()).decode()
    })


# In[7]:


pd.DataFrame(result).to_csv('submission.csv', index=None)

