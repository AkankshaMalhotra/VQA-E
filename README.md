# Visual QA Assistance for Visual Impairment

In recent years, with the advent of deep learning, the AI community is trying to move towards multidis- ciplinary approach where we combine various facets of AI like natural language processing, computer vision, reinforcement learning, etc. One such example is Visual Question Answering (VQA) which combines Natural Language Processing and Computer Vision. Several such VQA models have been proposed in recent years, which combine textual information and visual data, that allow user interaction in the form of natural language question-answers. While these works have a more general task, we aim to use this model to assist those who suffer from visual impairments. We implemented a VQA model that uses a soft-guided question co-attention mechanism to explain both the image and the question attention. The model was made 'explainable' through visual and textual modalities, by retrieving a complementary image to the given input image and a text explanation as 'explanation-by-elaboration', to justify the predicted answer.

- In this project we combined the ```VQA v2``` and the ```VQA-E``` dataset to create a custom ```VQA-EI``` dataset that aims to train 
and provide two modalities of explanations for the model output.
- We reported 70% accuracy on the answer predictions using GRU model using Glove embedding, and a ResNet152 pre-trained network for the 
image embeddings, for 20 epochs and 1000 data points.

VQA-EI dataset: https://drive.google.com/file/d/1JHKaF-0aZEVdOodrn8bNw62WksANG2y3/view?usp=sharing

supporting data:
- [MS-COCO images](http://images.cocodataset.org/zips/train2014.zip)  

- [Glove](http://nlp.stanford.edu/data/glove.6B.zip)

[Paper](Deep_Learning_Project.pdf)
