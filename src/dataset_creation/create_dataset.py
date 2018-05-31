import os
from tqdm import tqdm
import PIL.Image as Image
import pickle
import numpy as np

def get_triple(path_n, image_names, start, end):
    result = list()
    for i in range(start, end + 1, 1):
        image = Image.open('{}/{}'.format(path_n, image_names[i]))
        result.append(np.array(image))
    return np.array(result)


if __name__ == '__main__':
    image_path = 'images'
    image_name = os.listdir(image_path)
    dataset = list()
    start = 0
    end = 0
    idx = int(len(image_name)/15)

    for i in tqdm(range(0, len(image_name)-3, 3)):
        example = get_triple(image_path, image_name, i, i+3)
        with open('dataset/batch_{}_len_{}_.pkl'.format(len(dataset), start), 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            start += 1
    #pickle.dump('dataset.pkl')