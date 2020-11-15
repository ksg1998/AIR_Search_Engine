import pandas as pd
import os
import sys
import pickle
import math
#uncomment the next two lines when running it for the first time
#import nltk
#nltk.download('wordnet')

#import nltk
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
#from pycontractions import Contractions
import pandas as pd
import string

def preprocess(s):
    s = [i for i in s if str(i) != 'nan']
    s = ''.join(s)
    s = s.lower()
    s = s.replace('https://archive.org/details/', '')
    s = s.replace('.jpg', '')
    s = s.replace('.thumbs', '')
    s = s.replace('/', ' ')
    s = s.replace('_', ' ')
    s = s.replace('#', ' ')
    s = s.replace('  ', ' ')
    s = s.translate(translator)
    s = s.strip()

    tokens = word_tokenize(s)
    tokens = [i for i in tokens if not i in stop_words]
    #tokens = [stemmer.stem(i) for i in tokens]
    #tokens = [lemmatizer.lemmatize(i, pos ='v') for i in tokens]
    tokens = [lemmatizer.lemmatize(i) for i in tokens]
    return tokens

def write_to_disk(tree, filename):
    f = open(filename, "wb")
    pickle.dump(tree, f)
    f.close()

def create_dict():
    filenames = os.listdir(os.curdir+"/TelevisionNews/")
    filenames=[os.curdir+"/TelevisionNews/"+i for i in filenames]
    word_dict = {}
    totalnorows = 0
    for file in filenames:
        df = pd.read_csv(file)
        #print(file)

        for row in range(len(df)):
            totalnorows += 1
            token = preprocess(list(df.iloc[row]['URL']))
            # token = preprocess(list(df.iloc[row].values)) --> for all columns
            for i in range(len(token)):
                if token[i] not in word_dict:
                    word_dict[token[i]] = [1,{file:[1,{row:[i]}]}] #{word:[freqinCorpus,{filename:[freqOccurenceInFile,{rowno:[position]}]}]}
                else:
                    word_dict[token[i]][0]+=1
                    if file not in word_dict[token[i]][1]:
                        word_dict[token[i]][1][file]=[1,{row:[i]}]
                    else:
                        word_dict[token[i]][1][file][0]+=1
                        if row not in word_dict[token[i]][1][file][1]:
                            word_dict[token[i]][1][file][1][row]=[i]
                        else:
                            word_dict[token[i]][1][file][1][row].append(i)
                            
    
    #Updating dict
    #Should we use tf-idf? Do we consider documents in same csv file or across?
    
    for word in word_dict:
        idf = 0
        for csvfile in word_dict[word][1]:
            idf += len(word_dict[word][1][csvfile][1])#For every csv file, get #rows in which this term appears
        inverse = math.log(totalnorows/(idf+1)) + 1
        for csvfile in word_dict[word][1]:
            for rowno in word_dict[word][1][csvfile][1]:
                tf = len(word_dict[word][1][csvfile][1][rowno])
		tf = math.sqrt(tf)
                value = tf * inverse
                word_dict[word][1][csvfile][1][rowno] = [word_dict[word][1][csvfile][1][rowno],value]
                #{word:[freqinCorpus,{filename:[freqOccurenceInFile,{rowno:[[position],tfidf]}]}]}              
    
    #print("Length of dictionary: ",len(word_dict))
    #print("Size of dictionary: ",sys.getsizeof(word_dict))
    #print(word_dict["hello"])
    write_to_disk(word_dict, "snippet_index.txt")
    print("Written to disk")
    
        
if __name__ == "__main__":
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    #stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()
    translator = str.maketrans('', '', string.punctuation)
    create_dict()




