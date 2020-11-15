import pandas as pd
import os
import sys
import pickle
#uncomment the next two lines when running it for the first time
import nltk
nltk.download('wordnet')

import nltk
nltk.download('punkt')
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

class Node:
    def __init__(self, data, parent = None):
        self.data = list([data])
        self.parent = parent
        self.child = list()
    
    def __lt__(self, node):
        return self.data[0] < node.data[0]
    
    def _isLeaf(self):
        return len(self.child) == 0

    def _add(self, node):
        for child in node.child:
            child.parent = self
        self.data = self.data+node.data
        self.data.sort()
        self.child = self.child+node.child
        if len(self.child) > 1:
            self.child.sort()
        if len(self.data) > 2:
            self._rearrange()

    def _insert(self, node):
        if self._isLeaf():
            self._add(node)
        elif node.data[0] > self.data[-1]:
            self.child[-1]._insert(node)
        else:
            for i in range(0, len(self.data)):
                if node.data[0] < self.data[i]:
                    self.child[i]._insert(node)
                    break

    def _rearrange(self):
        l_child = Node(self.data[0], self)
        r_child = Node(self.data[2], self)
        if self.child:
            self.child[0].parent = l_child
            self.child[1].parent = l_child
            self.child[2].parent = r_child
            self.child[3].parent = r_child
            l_child.child = [self.child[0], self.child[1]]
            r_child.child = [self.child[2], self.child[3]]
        self.child = [l_child]
        self.child.append(r_child)
        self.data = [self.data[1]]
        if self.parent:
            if self in self.parent.child:
                self.parent.child.remove(self)
            self.parent._add(self)
        else:
            l_child.parent = self
            r_child.parent = self

    def _find(self, item):
        if item in self.data:
            return True
        elif self._isLeaf():
            return False
        elif item > self.data[-1]:
            return self.child[-1]._find(item)
        else:
            for i in range(len(self.data)):
                if item < self.data[i]:
                    return self.child[i]._find(item)
            
    def _inorder(self,inorderlist):
        if len(self.child)>2:
            self.child[0]._inorder(inorderlist)
            inorderlist.append(self.data[0])
            self.child[1]._inorder(inorderlist)
            inorderlist.append(self.data[1])
            self.child[2]._inorder(inorderlist)
            
        elif len(self.child)>1:
            self.child[0]._inorder(inorderlist)
            for j in self.data:
                inorderlist.append(j)
            self.child[1]._inorder(inorderlist)
        else:
            for i in self.data:
                inorderlist.append(i)
                
    def _findprefixwords(self,limits,prefixwordlist):
        if len(self.child)>2:
            self.child[0]._findprefixwords(limits,prefixwordlist)
            if (self.data[0] >= limits[0] and self.data[0] < limits[1]):
                prefixwordlist.append(self.data[0])
            self.child[1]._findprefixwords(limits,prefixwordlist)
            if (self.data[1] >= limits[0] and self.data[1] < limits[1]):
                prefixwordlist.append(self.data[1])
            self.child[2]._findprefixwords(limits,prefixwordlist)
            
        elif len(self.child)>1:
            self.child[0]._findprefixwords(limits,prefixwordlist)
            for j in self.data:
                if (j >= limits[0] and j < limits[1]):
                    prefixwordlist.append(j)
            self.child[1]._findprefixwords(limits,prefixwordlist)
        else:
            for i in self.data:
                if (i >= limits[0] and i < limits[1]):
                    prefixwordlist.append(i)
                  
        
            
class Tree:
    def __init__(self):
        self.root = None

    def insert(self, item):
        if self.root is None:
            self.root = Node(item)
        else:
            self.root._insert(Node(item))
            while self.root.parent:
                self.root = self.root.parent
        return True

    def find(self, item):
        if self.root is None:
            return False
        return self.root._find(item)
        
    def inorder(self,inorderlist):
        print("Inorder traversal: ")
        self.root._inorder(inorderlist)
        
    def findprefixwords(self,limits,prefixwordlist):
        print("Found words: ")
        self.root._findprefixwords(limits,prefixwordlist)
        

    
if __name__=="__main__":
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    #stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()
    translator = str.maketrans('', '', string.punctuation)
    
    tree = Tree()
    filenames = os.listdir(os.curdir+"/TelevisionNews/")
    filenames=[os.curdir+"/TelevisionNews/"+i for i in filenames]
    word_dict = {}
    for file in filenames:
        df = pd.read_csv(file)
        print(file)
        for row in range(len(df)):
            token = preprocess(list(df.iloc[row]['Snippet']))
            # token = preprocess(list(df.iloc[row].values)) --> for all columns
            for i in range(len(token)):
                if not tree.find(token[i]):
                    tree.insert(token[i])
   
    print("Size of Tree: ",sys.getsizeof(tree))
    write_to_disk(tree, "twothreetree.txt")
    print("Written to disk")  
    
    