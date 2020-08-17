#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os

import numpy as np
import pandas as pd

from skimage.draw import polygon2mask
from skimage.io import imsave, imshow

from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")


# In[2]:


meta = [i for i in os.listdir('train/train/') if '.json' in i]


# In[3]:


os.mkdir('mask')


# In[4]:


df = []

for i in tqdm(meta):
    image_meta = json.load(open(f'train/train/{i}'))
    shapes = image_meta['shapes']
    label = shapes[0]['label']
    for num, shape in enumerate(shapes):
        if shape['shape_type'] not in ['polygon', 'rectangle']:
            raise Exception(shape['shape_type']+' '+image_name)
        x_min, y_min = min(shape['points'], key=lambda x: x[0])[0], min(shape['points'], key=lambda x: x[1])[1]
        x_max, y_max = max(shape['points'], key=lambda x: x[0])[0], max(shape['points'], key=lambda x: x[1])[1]
        
        image_shape = (image_meta['imageHeight'], image_meta['imageWidth'])
        polygon = np.array([[i[1], i[0]] for i in shape['points']])
        mask = polygon2mask(image_shape, polygon)
        mask_name = i.replace('.json', '') + '_' + str(num)
        imsave(f'mask/{mask_name}.png', mask.astype(int))
        
        df.append({
            'ImageID': i.replace('.json', ''),
            'MaskName': i.replace('.json', '') + '_' + str(num),
            'Label': shape['label'],
            'XMin': x_min / image_meta['imageWidth'],
            'YMin': y_min / image_meta['imageHeight'],
            'XMax': x_max / image_meta['imageWidth'],
            'YMax': y_max / image_meta['imageHeight'],
        })


# In[5]:


df = pd.DataFrame(df)


# In[6]:


mapper = {
    'staphylococcus_aureus': 1,
    'staphylococcus_epidermidis': 2,
    'ent_cloacae': 3,
    'c_kefir': 4,
    'moraxella_catarrhalis': 5,
    'klebsiella_pneumoniae': 6
}


# In[7]:


df['LabelName'] = df.Label.map(mapper)


# In[8]:


df.to_csv('full_meta.csv', index=None)


# In[9]:


image_label_df = df[['ImageID', 'Label']].drop_duplicates()


# In[10]:


train_images, val_images, _, _ = train_test_split(image_label_df.ImageID, image_label_df.Label, test_size=0.1)


# In[11]:


train = df[df.ImageID.isin(train_images)]


# In[12]:


train.to_csv('train.csv', index=None)


# In[13]:


val = df[df.ImageID.isin(val_images)]


# In[14]:


val.to_csv('val.csv', index=None)


# In[ ]:




