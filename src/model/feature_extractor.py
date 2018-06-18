from src.model.youtube_8m.feature_extractor import feature_extractor
import pickle
import os

def get_features(image):
    return feature_extractor(image)



'''
path = '../../../dataset'
print(os.listdir(path))

files = os.listdir(path)
file_path = os.path.join(path, files[0])
print(file_path)
'''

def extract_features(image):
    extracor = feature_extractor.YouTube8MFeatureExtractor()
    return extracor.extract_rgb_frame_features(image)


def get_image_patch(file_path):
    p = None
    with open(file_path, 'rb') as f:
        p = pickle.load(f)
    return p
