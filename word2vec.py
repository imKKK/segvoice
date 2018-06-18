import numpy as np
import pickle
from scipy.spatial.distance import cosine

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def read_text(fname):
    in_word = False
    word = ''
    vocab = {}
    text = []
    id = 0
    for line in open(fname,'rt'):
        
        for c in line:
            c = c.lower()

            if c>='a' and c<='z':
                if in_word == False:
                    in_word = True 
                word += c
            else:
                if in_word == True:
                    if len(word) > 2 and len(word) < 16:
                        if vocab.get(word) == None:
                            vocab[word] = [id,1]
                            id += 1
                        else:
                            vocab[word][1] +=1
                         
                        if vocab[word][1] < 5000:
                            text.append(vocab[word][0])

                    word = ''     

    return vocab,text

def distance(word):
    with open('model.pkl','rb') as f:
        matrix = pickle.load(f)
    with open('vocab.pkl','rb') as f:
        vocab = pickle.load(f)
    scores = {}
    for k,v in vocab.items():
        scores[k] = 1-cosine(matrix[v[0]],matrix[vocab[word][0]])
    top10 = sorted(scores.items(),key = lambda x:x[1])[-10:]
    return top10

def compute(fname):
    dim = 100
    delta = 0.02
    win = 5
    neg_size = 5
    vocab,text = read_text(fname)
    matrix = np.random.uniform(-.5/dim,.5/dim,(len(vocab),dim))
    sample = np.zeros((win*2+neg_size,dim))

    for epoch in range(3):
        for i in range(win,len(text)-win):
            if i % 100000 == 0:
                print(i,'/',len(text))

            pos = text[i-win:i]+text[i+1:i+win+1]

            neg = []

            for j in range(neg_size):
                while True:
                    id = np.random.randint(len(vocab))
                    if not id in pos:
                            neg.append(id)
                            break  

            for p in range(win*2):
                sample[p] = matrix[pos[p]]

            for n in range(neg_size):
                sample[n+win*2] = matrix[neg[n]]

            #forward
            out = sigmoid(np.dot(sample,matrix[text[i]].transpose()))
            out *= (1-out)
            out[win*2:] *= -1
          
            #backward
            for j in range(len(pos)):
                matrix[pos[j]] -= delta*out[j]*matrix[text[j]]

            for j in range(len(neg)):
                matrix[neg[j]] -= delta*out[j+win*2]*matrix[text[j]] 

    with open('vocab.pkl','wb') as f:
        pickle.dump(vocab,f)

    with open('model.pkl','wb') as f:
        pickle.dump(matrix,f)    

compute('enwiki.txt')
