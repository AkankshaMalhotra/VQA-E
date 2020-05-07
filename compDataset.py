import json
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np


dataDir = './Train/'

dataFiles = {
    'Validation': {
        'imageFile': 'val2014/',
        'annotationFile': 'v2_mscoco_val2014_annotations.json',
        'questionFile': 'v2_OpenEnded_mscoco_val2014_questions.json',
        'compListFile': 'v2_mscoco_val2014_complementary_pairs.json',
        'explanationFile': 'VQA-E_val_set.json'
    },
    'Train': {
        'imageFile': 'train2014/',
        'annotationFile': 'v2_mscoco_train2014_annotations.json',
        'questionFile': 'v2_OpenEnded_mscoco_train2014_questions.json',
        'compListFile': 'v2_mscoco_train2014_complementary_pairs.json',
        'explanationFile': 'VQA-E_train_set.json'
    }
}

dataType = 'Train'


# load data files
def loadData(path, dataType):

    # get images
    image_file = path + dataFiles[dataType]['imageFile']
    _images = [img for img in listdir(image_file) if isfile(join(image_file, img))]
    
    # get annotations
    annotation_file = path + dataFiles[dataType]['annotationFile']
    _annotations = json.load(open(annotation_file, 'r'))['annotations']
    
    # get questions
    question_file = path + dataFiles[dataType]['questionFile']
    _questions = json.load(open(question_file, 'r'))['questions']
    
    # get complementary pairs list
    comp_pair_file = path + dataFiles[dataType]['compListFile']
    _comp_pairs = json.load(open(comp_pair_file, 'r'))
    
    # get vqa-e dataset
    explanation_file = path + dataFiles[dataType]['explanationFile']
    _explanations = json.load(open(explanation_file, 'r'))
    
    return _images, _annotations, _questions, _comp_pairs, _explanations

images, annotations, questions, comp_pairs, explanations = loadData(dataDir, dataType)

print(f'Images: {len(images)}')
print(f'Annotations: {len(annotations)}')
print(f'Questions: {len(questions)}')
print(f'Complementary Pairs: {len(comp_pairs)}')
print(f'Explanations: {len(explanations)}')


# Dataset with Complementary Images

dataset = []

for data in tqdm(explanations):
    # get data fields
    imgId = data['img_id']
    question = data['question']
    
    # get question_id_1 from input imgId and question
    img_question = list(filter(lambda x:x['image_id']==imgId and x['question']==question,questions))
    question_id_1 = img_question[0]['question_id']

    # get complementary pair question_id_2
    for pair in comp_pairs:
        if question_id_1 in pair:
            question_id_2 = list(filter(lambda x : x != question_id_1, pair))[0]

    # get image_id from questions with question_id == question_id_2
    comp_question = list(filter(lambda x:x['question_id']==question_id_2,questions))
    comp_img_id = comp_question[0]['image_id']
    
    # copy old fields
    data2 = data.copy()
    # add new field
    data2['comp_img_id'] = comp_img_id
    
    dataset.append(data2)


# save dataset
with open(dataDir + dataType + '/VQA-E-C.json', 'w') as f:
    json.dump(dataset, f)
