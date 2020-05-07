import tensorflow as tf
import numpy as np
import pickle
import multiprocessing

dataDir = './Train/'

# load image embedding vectors
filename = dataDir + 'image_vectors_resnet152'
with open(filename, 'rb') as f:
    processed_images = pickle.load(f)

img_embeddings = np.array([values for key, values in processed_images.items()])
img_embeddings_chunks = img_embeddings.copy()

# Euclidean norm of each feature vector in vector embeddings with other vectors (82788x82783)

manager = multiprocessing.Manager()
euclidean_norm_matrix = manager.list()


def createMatrix(img_embd):

	for img_vec in img_embeddings:
		euclidean_norm_matrix.append(np.linalg.norm(img_vec - img_embd))

if __name__ == '__main__':
    pool1 = multiprocessing.Pool(processes=96) # Instantiate the pool of workers

    pool1.map(createMatrix, img_embeddings_chunks, 1000)
    pool1.close()
    pool1.join()

    print(len(euclidean_norm_matrix))

    # save dataset
    with open('embedding_norm_matrix_euclidean.pickle', 'wb') as f:
        pickle.dump(list(euclidean_norm_matrix), f)
