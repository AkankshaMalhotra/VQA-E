import json
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np
import multiprocessing


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

    # get questions
    question_file = path + dataFiles[dataType]['questionFile']
    _questions = json.load(open(question_file, 'r'))['questions']
    
    # get complementary pairs list
    comp_pair_file = path + dataFiles[dataType]['compListFile']
    _comp_pairs = json.load(open(comp_pair_file, 'r'))
    
    # get vqa-e dataset
    explanation_file = path + dataFiles[dataType]['explanationFile']
    _explanations = json.load(open(explanation_file, 'r'))
    
    return _questions, _comp_pairs, _explanations

questions, comp_pairs, explanations = loadData(dataDir, dataType)

manager = multiprocessing.Manager()
dataset = manager.list()
missing = manager.list()

print(f'Questions: {len(questions)}')
print(f'Complementary Pairs: {len(comp_pairs)}')
print(f'Explanations: {len(explanations)}')


# Dataset with Complementary Images

def createDataset(data):
    # get data fields
    imgId = data['img_id']
    question = data['question']

    # get question_id_1 from input imgId and question
    img_question = list(filter(lambda x:x['image_id']==imgId and x['question']==question,questions))
    question_id_1 = img_question[0]['question_id']

    question_id_2 = -1
    # get complementary pair question_id_2
    for pair in comp_pairs:
        if question_id_1 in pair:
            question_id_2 = list(filter(lambda x : x != question_id_1, pair))[0]

    # get image_id from questions with question_id == question_id_2
    print('question_id_2: ', question_id_2)
    if question_id_2 != -1:
        comp_question = list(filter(lambda x:x['question_id']==question_id_2,questions))
        comp_img_id = comp_question[0]['image_id']

        # copy old fields
        data2 = data.copy()
        # add new field
        data2['comp_img_id'] = comp_img_id

        dataset.append(data2)
    else:
        missing.append(question_id_1)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=96) # Instantiate the pool here

    pool.map(createDataset, explanations, 1000)
    pool.close()
    pool.join()
    print(len(dataset))

    # save dataset
    with open('vqa-e.json', 'w') as f:
        json.dump(list(dataset), f)

    # save missing question_id_1
    with open('missing.json', 'w') as f1:
        json.dump(list(missing), f1)    
