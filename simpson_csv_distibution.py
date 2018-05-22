import pandas as pd
from PIL import Image

bbox_datas = pd.read_csv('./simpson/annotation.txt',header=None)
bbox_datas_df = pd.DataFrame(bbox_datas)

#Changes the order of columns and sets the name.
bbox_datas_df = bbox_datas_df[[0,5,1,2,3,4]]
bbox_datas_df.columns = ['filename','class','xmin','ymin','xmax','ymax']
bbox_datas_df = bbox_datas_df.sort_values(['class'])
bbox_datas_df = bbox_datas_df.reset_index(drop=True)

class_simpson = bbox_datas_df.drop_duplicates('class')['class'].values.tolist()

result = pd.DataFrame(columns=('filename','width','height','class','xmin','ymin','xmax','ymax'))
cnt = 0
class_ = 0

for i in range(len(bbox_datas_df)):

    bbox_data = bbox_datas_df.ix[i]
    if (class_simpson[class_] == bbox_data['class'] and cnt < 160):
        if (bbox_data['xmax'] - bbox_data['xmin'] > 30 and bbox_data['ymax'] - bbox_data['ymin'] > 30):
            cnt += 1
            filename = bbox_data['filename']
            filename = filename.replace('characters', 'simpson')
            filename = filename.replace('simpson2', 'simpson')

            im = Image.open(filename)
            width, height = im.size

            result.loc[i] = ({
                'filename': filename,
                'width': width,
                'height': height,
                'class': bbox_data['class'],
                'xmin': bbox_data['xmin'],
                'ymin': bbox_data['ymin'],
                'xmax': bbox_data['xmax'],
                'ymax': bbox_data['ymax']
            })
        else:
            print(bbox_data['filename'])
    else:
        if (class_simpson[class_] != bbox_data['class']):
            if class_ < len(class_simpson)-1:
                cnt = 0
                class_ += 1
            else:
               break

rate = len(result)/8.0
result = result.sample(frac=1)
train_df = result[int(rate):]
validate_df = result[:int(rate)]
train_df.to_csv('./dataset/train.csv',index=None)
validate_df.to_csv('./dataset/validate.csv',index=None)
