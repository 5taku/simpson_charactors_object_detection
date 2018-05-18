import pandas as pd
from PIL import Image
from random import shuffle

bbox_datas = pd.read_csv('./simpson/annotation.txt',header=None)
bbox_datas_df = pd.DataFrame(bbox_datas)

bbox_datas_df = bbox_datas_df[[0,5,1,2,3,4]]
bbox_datas_df.columns = ['filename','class','xmin','ymin','xmax','ymax']

result = pd.DataFrame(columns=('filename','width','height','class','xmin','ymin','xmax','ymax'))

for i in range(len(bbox_datas_df)):
    bbox_data = bbox_datas_df.ix[i]

    filename = bbox_data['filename']
    filename = filename.replace('characters','simpson')
    filename = filename.replace('simpson2', 'simpson')

    im = Image.open(filename)
    width,height = im.size

    result.loc[i] = ({
      'filename':filename,
      'width':width,
      'height':height,
      'class':bbox_data['class'],
      'xmin':bbox_data['xmin'],
      'ymin':bbox_data['ymin'],
      'xmax':bbox_data['xmax'],
      'ymax':bbox_data['ymax']
    })

rate = len(result)/8.0
result = result.sample(frac=1)
train_df = result[int(rate):]
validate_df = result[:int(rate)]
train_df.to_csv('./dataset/train.csv',index=None)
validate_df.to_csv('./dataset/validate.csv',index=None)
