word2vec_homework


#######################################
#######################################
#
#@@@@@Dependencies import@@@@@
#
#######################################

#import functions from old python versions
from __future__ import absolute_import, division, print_function

#for word encoding
import codecs
#to search through corpus data (regex)
import glob
#for debugging
import logging
#concurrency (speed up model)
import multiprocessing
#operating system functions (e.g., read files)
import os
#printing
import pprint
#regular expressions
import re

#nlp toolkit
import nltk
#word2vec
import gensim.models.word2vec as w2v
#dimensionality reduction; visualize data
import sklearn.manifold
#linear algebra
import numpy as np
#plotting
import matplotlib.pyplot as plt
#pandas (pd)
import pandas as pd
#visualization
import seaborn as sns

import plotly







#######################################
#######################################
#
#@@@@@Import Data@@@@@
#
#######################################
train_filelist = []
# Extract all file names for training data
# Append them all to list for looping
for file_name in os.listdir("/Users/simone/desktop/dataHomework1/training-monolingual_tokenized_shuffled"):
    train_filelist.append(file_name)
train_filelist.remove('.DS_Store')


# Run this to select a sub-sample of the training set 
# This will reduce computational cost of future steps
import random
# Select amount of files out of total
n = 5
random.seed(101)
# Randomly select n files from list (random.seed() for replicability)
# I already ran the code and the seed selected the following files
train_filelist = random.sample(train_filelist, n)


#find the absolute path to folder, (pwd terminal command)
#Convert text into readable utf-8 file format and store into a single variable
corpus_train = u""
counter = len(train_filelist)
counterRead = int(1)
for train_file in train_filelist:
    print("Reading " + str(counterRead) + " of " + str(counter))
    print("Reading '{0}'...".format(train_file))
    with open(str("/Users/simone/desktop/dataHomework1/training-monolingual_tokenized_shuffled/" + train_file),"rb") as file:
        file_content = file.read()
        text = file_content.decode('utf-8')
        corpus_train += text
    print("Corpus is now {0} characters long".format(len(corpus_train)))
    print()
    counterRead = counterRead + 1







#######################################
#######################################
#
#@@@@@Clean Data@@@@@
#
#######################################

#Tokenize order 
#first: token=sentence || second: token=word

#Tokenize the raw data into sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_train)

# Tokenize the raw data into words
# & Remove some unnecessary linguistic features
# [^a-zA-Z] returns all non-letters e.g. split hyphens, remove punctuations
# The .split method separates all words into tokens, splitting at each space
def sentence_to_word_function(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

Training_Tokens = []
for sentencetoken in raw_sentences:
    if len(sentencetoken) > 0:
        Training_Tokens.append(sentence_to_word_function(sentencetoken))

#Make all tokens lowercase
Training_Tokens_lower = [[x.casefold() for x in sublst] for sublst in Training_Tokens]

#Remove stopwords tokens so that context window captures semantic information better
#commented out because homework requires calculation of similarity between the stopword 'she' and other vectors
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words('english'))
#new_suffixes = {"'s", "'t", "'ve", "'d", "'ll", "'re"}
#stop_words.update(new_suffixes)
#Training_Tokens_Clean = [[wordx for wordx in sublst if wordx not in stop_words] for sublst in Training_Tokens_lower]


#Check amount of tokens in training corpus
token_count = sum([len(Training_Tokens_lower) for Training_Tokens_lower in Training_Tokens_lower])
print("The training corpus contains {0:,} tokens".format(token_count) + " including stopwords")
#Check how a sentence looks across various various stages of cleaning
print(raw_sentences[5])
print(Training_Tokens_lower[5])
print(Training_Tokens[5])
#print(Training_Tokens_Clean[5])








#######################################
#######################################
#
#@@@@@WORD2VEC Hyperparameters Setup@@@@@
#
#######################################

# set number of features in each word embedding
# higher dimensionality is good for accuracy but bad for computational cost 
# find a balance between accuracy of model and computational cost of training
num_features = 300

# Minimum word count threshold for word to be included in training
# Word2vec ignores all words with frequency lower than this threshold
# This removes noise due to infrequent words, with too few exemplars in data
min_wcount = 3

# Number of threads to run in parallel
# Get the amount of CPU's to assign to the task
num_workers = multiprocessing.cpu_count()

# Context window length
context_size = 2

# Downsample setting for frequent words
# Make it so that high frequency words aren't continuosly re-used in training 
# Number between 0 - 1e-5 should be good
downsampling = 1e-3

# Seed for the random number generator
# rand num generator picks the sections of text to train over
seed = 1

#Select between CBOW (0) or SkipGram (1)
modeltype = 0


#######################################
#######################################
#
#@@@@@WORD2VEC Training@@@@@
#
#######################################

#Build the model
word2vecTrained = w2v.Word2Vec(
    seed=seed,
    workers=num_workers,
    vector_size=num_features,
    min_count=min_wcount,
    window=context_size,
    sample=downsampling,
    sg=modeltype
)

#Build the vocabulary from the corpus
#This is a list of all unique tokens in the corpus
#word2vecTrained.build_vocab(Training_Tokens_Clean)
word2vecTrained.build_vocab(Training_Tokens_lower)
vocabW2V = word2vecTrained.wv.index_to_key
print("Word2Vec vocabulary length:", len(word2vecTrained.wv.index_to_key))
#check a few tokens
list(vocabW2V)[0:60]

#check that words needed for homework are in the vocabulary (in the corpus)
if 'she' in vocabW2V:
    print('she is present in the list')
else:
    print('she NOT present in the list') 
if 'like' in vocabW2V:
    print('like is present in the list')
else:
    print('like NOT present in the list') 
if 'cat' in vocabW2V:
    print('cat is present in the list')
else:
    print('cat NOT present in the list') 


#Train the model
#word2vec will iteratively train on the data for the number of epochs selected
#each time it will run through the sentences and predict a given word by its context words
#the context words will be the ones surrounding the target word but within the boundaries of the 
#context window. Each iteration will result in a predicted output, and a loss will be calculated by
#comparing the predicted output to the actual output (i.e., the labels)
#the loss function will then be used to update the word embeddings, which are vector
#representations of each word. After 50 epochs we are left with embeddings that should represent
#the semantic value (i.e., meaning) of each word in a 300 dimensional space, because we selected 300 as a dimensionality
word2vecTrained.train(Training_Tokens_lower, total_examples=word2vecTrained.corpus_count, epochs=50)

#Save trained model
#if not os.path.exists("trained"):
#    os.makedirs("trained")
#word2vecTrained.save(os.path.join("trained", "word2vecTrained.w2v"))
#load
#word2vecTrained = w2v.Word2Vec.load(os.path.join("trained", "word2vecTrained.w2v"))







#######################################
#######################################
#
#@@@@@Reduce dimensionality/Visualize/Play with vectors@@@@@
#
#######################################

####Run cosine similarity analysis
# computes the cosine angle between the target vector and all other embeddings
# returns the top 10 embeddings that are closest to the target vector
# a high cosine similarity implies a high semantic similarity between the two words


word2vecTrained.wv.most_similar("cat")
#not including stop words in training 
# [('cats', 0.49014317989349365), ('dog', 0.4560866057872772), ('bird', 0.3981786370277405), 
# ('moose', 0.3798355162143707), ('birds', 0.37671682238578796), ('monkey', 0.3718675971031189), 
# ('kitten', 0.37163063883781433), ('faeces', 0.3709704875946045), ('domesticated', 0.3697093427181244), ('dogs', 0.3677574396133423)]
##########including stop words in training
# [('dog', 0.4892568290233612), ('reptile', 0.446358859539032), ('cats', 0.43958577513694763), 
# ('dachshund', 0.416104793548584), ('raccoon', 0.4108794927597046), ('baby', 0.4103401303291321), 
# ('pigeons', 0.400519460439682), ('rottweiler', 0.3929516673088074), 
# ('hummingbirds', 0.38905084133148193), ('droppings', 0.3885918855667114)]



word2vecTrained.wv.most_similar("she")
#not including stop words in training, no embedding for 'she'
##########including stop words in training
# [('he', 0.9543355703353882), ('it', 0.7333446145057678), ('i', 0.649720311164856), 
# ('nobody', 0.6326048374176025), ('they', 0.6292216181755066), ('palin', 0.5377972722053528), 
# ('we', 0.5216145515441895), ('never', 0.5201559662818909), 
# ('favre', 0.5084103941917419), ('everybody', 0.5076685547828674)]




word2vecTrained.wv.most_similar("like")
#not including stop words in training 
# [('unlike', 0.5025439858436584), ('know', 0.47939589619636536), ('prefer', 0.4520551264286041), 
# ('think', 0.4471327066421509), ('crazy', 0.42439866065979004), ('really', 0.4224821627140045), 
# ('different', 0.41288119554519653), ('tend', 0.41089633107185364), ('want', 0.4105139374732971), ('see', 0.4102248549461365)]

#including stop words
#[('prefer', 0.508956253528595), ('mean', 0.44698867201805115), ('akin', 0.429590106010437), 
# ('unlike', 0.42351430654525757), ('resembling', 0.40809166431427), ('know', 0.3994082510471344), 
# ('remember', 0.3961983621120453), ('enjoy', 0.3948531150817871), 
# ('think', 0.39134809374809265), ('resemble', 0.38888171315193176)]



#####@@@@@@@@########@@@@@@@@########@@@@@@@@@########@@@@@@@@@
#####@@@@@@@@########@@@@@@@@########@@@@@@@@@########@@@@@@@@@


#define dimensionality reduction function with sklearn, t-SNE method
def dimension_reduction(model):
    dimensions = 2  # final number of dimensions/features for each embedding

    # from model extract unique words & embeddings as numpy arrays
    vectorsW2V = np.asarray(model.wv.vectors)
    labelsW2V = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # dimensionality reduction using t-SNE method from sklearn
    tsne = sklearn.manifold.TSNE(n_components=dimensions, random_state=0)
    vectors = tsne.fit_transform(vectorsW2V)

    xvalues = [i[0] for i in vectors]
    yvalues = [i[1] for i in vectors]
    return xvalues, yvalues, labelsW2V

# define dimensionality reduction function with sklearn, truncated SVD method
from sklearn.decomposition import TruncatedSVD 
truncSVD=TruncatedSVD(2)
vectorsW2Vt = np.asarray(word2vecTrained.wv.vectors)
labelsW2Vt = np.asarray(word2vecTrained.wv.index_to_key)
#check that output looks correct, embeddings should contain 300 features
if vectorsW2Vt.shape[1] == 300:
    print('300 features in embeddings')

#run dimensionality reduction with truncated SVD
#SVD is a matrix factorization method that extracts the largest linear dependencies
#or eigenvectors, and utilizes those to reduce the dimensions of the original matrix
#eigenvectors with low eigenvalues are discarded
#SVD typically produces 3 matrices, but this function only outputs one of the orthogonal matrices 
X_truncated = truncSVD.fit_transform(vectorsW2Vt)
print(X_truncated.shape)
#run dimensionality reduction with t-SNE
#this is a probabilistic approach, which makes the assumption that the data
#follows the structure of a manifold, and position each point in space to match a neighbor
x_vals, y_vals, labels = dimension_reduction(word2vecTrained)
type(x_vals)
print(len(x_vals))

#define plot function with matplotlib
def plot2dTSNE(xvalue, yvalue, labelsW2V):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)
    plt.figure(figsize=(8, 8))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.show()

#define plot function with matplotlib SVD
def plot2dSVD(vectorsW2V, labelsW2V):
    import matplotlib.pyplot as plt
    import random

    x_vals = list(vectorsW2V[:, 0])
    y_vals = list(vectorsW2V[:, 1])
    random.seed(0)
    plt.figure(figsize=(8, 8))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.show()


#########Plot all embeddings
plot2dTSNE(x_vals, y_vals, labels)
plot2dSVD(X_truncated, labelsW2Vt)

##########################
#######Combine in dataframe
type(labelsW2Vt)
points = pd.DataFrame(pd.np.column_stack((labelsW2Vt, X_truncated)), columns=["word", "x", "y"])
type(points)
points[0:10]





