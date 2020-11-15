import logging
import requests
import json
import time
from elasticsearch import Elasticsearch
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
from spellchecker import SpellChecker

#from pycontractions import Contractions
import pandas as pd
import string
import functools

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
translator = str.maketrans('', '', string.punctuation)
spell = SpellChecker()

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
            if self.data[0] > limits[1]:
                return
            self.child[1]._findprefixwords(limits,prefixwordlist)
            if (self.data[1] >= limits[0] and self.data[1] < limits[1]):
                prefixwordlist.append(self.data[1])
            if self.data[1] > limits[1]:
                return
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
        self.root._findprefixwords(limits,prefixwordlist)
        
        

def read_from_disk(filename):
    f = open(filename, "rb")
    tree = pickle.load(f)
    f.close()
    return tree

def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else '' for i in text])

def spell_check(s):
    words = s.split()
    snew = ''
    for word in words:
        if not spell.known([word]):
            snew = snew + ' ' + spell.correction(word)
        else:
            snew = snew + ' ' + word
    return snew

def getlimits(inp):
    limits = [inp]
    if inp == 'z'*len(inp):
        limits.append('{')
        return limits
    newletter = None
    lastletter = inp[-1]
    inp = list(inp)
    currord = ord(lastletter)
    if currord == 122:
        while currord == 122:
            inp.pop()
            lastletter = inp[-1]
            currord = ord(lastletter)
            
    newletter = chr(currord + 1)    
    inp[-1] = newletter
    inp = ''.join(inp)
    limits.append(inp)
    return limits      
    

def preprocess_input(s):
    s = [i for i in s if str(i) != 'nan']
    s = ' '.join(s)
    s = remove_non_ascii(s)
    s = s.lower()
    s = s.replace('/', ' ')
    s = s.replace('#', ' ')
    s = s.replace('  ', ' ')
    s = s.translate(translator)
    s = s.strip()
    tokens = word_tokenize(s)
    tokens = [i for i in tokens if not i in stop_words]
    return tokens

def printer(score_dict):
    count = len(score_dict)
    result = {"count":count,"row":[]}
    for i in range(min(100,len(score_dict))):
        contents = score_dict[i][0].split(',')
        filename = contents[0]
        rowid = int(contents[1])
        df = pd.read_csv(filename)
        obj = df.iloc[rowid]
        obj["tf-idf"] = score_dict[i][1]
        obj = obj.to_json()
        result["row"].append(obj)
    return result
    
def printer_phrase(score_dict):
    count = len(score_dict)
    result = {"count":count,"row":[]}
    for i in range(min(100,len(score_dict))):
        contents = score_dict[i].split(',')
        filename = contents[0]
        rowid = int(contents[1])
        df = pd.read_csv(filename)
        obj = df.iloc[rowid]
        obj = obj.to_json()
        result["row"].append(obj)
    return result
    #Return the score and the contents of the actual row
    
def get_tfidf(inp,word_d):
    d_query = {}
    for word in set(inp):
        tf = 0
        idf = 0
        if(word in word_d):
            idf = word_d[word][0]
            tf = inp.count(word)
        else:
            continue
        d_query[word] = (1 + math.log(tf))*idf
    return d_query

def length(docrow):
    contents = docrow.split(',')
    file = contents[0] 
    rowno = int(contents[1])
    return lengthdict[file][rowno]

    
def search_postings(inp,word_d,flag):
    #{word:[idf,{filename:[freqOccurenceInFile,{rowno:[[position],tfidf]}]}]} 
    score_dict = {}
    tfidf_inp = get_tfidf(inp,word_d) #Dictionary {word:score,...}
    for inputterm in inp:
        if inputterm in word_d:
            for csvfile in word_d[inputterm][1]:
                for rowno in word_d[inputterm][1][csvfile][1]:
                    key = csvfile+","+str(rowno)
                    if key in score_dict:
                        
                        score_dict[key] = score_dict[key] + (tfidf_inp[inputterm] * word_d[inputterm][1][csvfile][1][rowno][1])#Figure out how to add tf-idf scores
                        
                    else:
                        score_dict[key] = tfidf_inp[inputterm] * word_d[inputterm][1][csvfile][1][rowno][1]
                   
    #Length normalise
    if flag == 1:
        for docrow in score_dict:
            score_dict[docrow] = score_dict[docrow]/(length(docrow)+6)
    elif flag == 2:#only snippet query
        for docrow in score_dict:
            score_dict[docrow] = score_dict[docrow]/(length(docrow))
        
    
    #sort score_dict based on the value and return top5
    if len(score_dict)==0:
        return []
    score_dict = [(k, v) for k, v in score_dict.items()] 
    score_dict = sorted(score_dict,key = lambda x : x[1],reverse = True)
    return printer(score_dict)

def intersect(l):
    return list(functools.reduce(lambda a,b : set(a)&set(b),l))
 
def phrase(inp, word_d):
    resultlist = []
    for inputterm in inp:
        doclist = []
        if inputterm not in word_d:
            return []
        for csvfile in word_d[inputterm][1]:
            doclist.append(csvfile)
        resultlist.append(doclist)
    candidatedoclist = intersect(resultlist)
    
    if len(candidatedoclist)==0:
        print("Empty")
        return {}
    
    candidaterowlist = []
    for csvfile in candidatedoclist:
        resultlist = []
        for inputterm in inp:
            rowlist = []
            for rowno in word_d[inputterm][1][csvfile][1]:
                rowlist.append(rowno)
            resultlist.append(rowlist)
            
        templist = intersect(resultlist)
        candidaterowlist += [csvfile + "," + str(i) for i in templist]
        
    if len(candidaterowlist)==0:
        print("Empty")
        return {}
    
    finallist = []
    for item in candidaterowlist:
        positionlist = []
        contents = item.split(",")
        docid = contents[0]
        rowno = int(contents[1])
        for inputterm in inp:
            positionlist.append(word_d[inputterm][1][docid][1][rowno][0])
        #[[position for word 1],[word2]...[wordn]]
        flag = 1
        for position in positionlist[0]:
            i = 1
            for elements in positionlist[1:]:
                if position + i not in elements:
                    flag = 0
                    break
                else:
                    flag = 1
                i += 1
            
        if flag:
            finallist.append(item)
    return printer_phrase(finallist)
    
def prefix_query(inp, tree, word_d,flag):
    #call inorder, get the candidate list and call the function that lists the row from csv file
    q = inp.split('*')
    inp = q[0]
    limits = getlimits(inp)
    prefixwordlist = []
    tree.findprefixwords(limits,prefixwordlist)
    return search_postings(prefixwordlist,word_d,flag)

def getcandidates(inp,permuterm):
    #return list of candidate words to be filled
    inp = inp.split('*')
    inp = inp[1] + '$'
    term_list = []
    for tk in permuterm.keys():
        if tk.startswith(inp):
            term_list.append(permuterm[tk])
    return term_list
    
def suffix_query(inp, permuterm, word_d,flag):
    suffixwordlist = getcandidates(inp,permuterm)
    return search_postings(suffixwordlist,word_d,flag)
    
def mix_query(inp, permuterm, tree, word_d,flag):
    q = inp.split('*')
    limits = getlimits(q[0])
    prefixwordlist = []
    tree.findprefixwords(limits,prefixwordlist)
    suffixwordlist = getcandidates(inp,permuterm)
    wordlist = list(set(prefixwordlist) & set(suffixwordlist))
    return search_postings(wordlist,word_d,flag)
    
def querytype(inp):
    inp = inp.strip()
    if inp[0] == '"' and inp[-1] == '"':
        return 1
    if '*' in inp and '=' not in inp:
        q = inp.split('*')
        if q[0] != '' and q[1] != '':
            return 4
        if q[0] != '':
            return 2
        if q[1] != '':
            return 3
    if '=' in inp:
        q = inp.split('=')
        if q[0] == "URL":
            return 5
        if q[0] == "MatchDateTime":
            return 6
        if q[0] == "Station":
            return 7
        if q[0] == "Show":
            return 8
        if q[0] == "IAShowID":
            return 9
        if q[0] == "IAPreviewThumb":
            return 10
        if q[0] == "Snippet":
            return 11
        else:
            #print("Invalid column name")
            return -1
    else:
        return 0

def input_function(inp,worddict_all,worddict_url,worddict_matchdatetime,worddict_station,worddict_show,worddict_iashowid,worddict_iapreviewid,worddict_snippet,tree,permuterm):
    flag = 0 #0 indicates no need of length normalisation
    if querytype(inp) == 0:       #normal query
        flag = 1
        inp = inp.split(' ')
        inp = preprocess_input(inp)
        return search_postings(inp,worddict_all,flag)

    elif querytype(inp) == 1:       #phrase query
        inp = inp[1:-1]
        inp = inp.split(' ')
        return phrase(inp,worddict_all)

    elif querytype(inp) == 2:       #wildcard mon* query
        flag = 1
        return prefix_query(inp,tree,worddict_all,flag)

    elif querytype(inp) == 3:       #wildcard *mon query
        flag = 1
        return suffix_query(inp,permuterm,worddict_all,flag)

    elif querytype(inp) == 4:       #wildcard abc*def
        flag = 1
        return mix_query(inp,permuterm,tree,worddict_all,flag)

    else: #column specific
        if querytype(inp) == 5:       #url query
            q = inp.split('=')
            req = q[1]
            if(querytype(req)==0):
                req = req.split(' ')
                req = preprocess_input(req)
                return search_postings(req,worddict_url,flag)
            elif(querytype(req)==1):
                req = req[1:-1]
                req = req.split(' ')
                return phrase(req,worddict_url)
            elif(querytype(req)==2):
                return prefix_query(req,tree,worddict_url,flag)
            elif(querytype(req)==3):
                return suffix_query(req,permuterm,worddict_url,flag)
            elif(querytype(req)==4):
                return mix_query(req,permuterm,tree,worddict_url,flag)

        elif querytype(inp) == 6:       #matchdatetime query
            q = inp.split('=')
            req = q[1]
            if(querytype(req)==0):
                req = req.split(' ')
                req = preprocess_input(req)
                return search_postings(req,worddict_matchdatetime,flag)
            elif(querytype(req)==1):
                req = req[1:-1]
                req = req.split(' ')
                return phrase(req,worddict_matchdatetime)
            elif(querytype(req)==2):
                return prefix_query(req,tree,worddict_matchdatetime,flag)
            elif(querytype(req)==3):
                return suffix_query(req,permuterm,worddict_matchdatetime,flag)
            elif(querytype(req)==4):
                return mix_query(req,permuterm,tree,worddict_matchdatetime,flag)

        elif querytype(inp) == 7:       #station query
            q = inp.split('=')
            req = q[1]
            if(querytype(req)==0):
                req = req.split(' ')
                req = preprocess_input(req)
                return search_postings(req,worddict_station,flag)
            elif(querytype(req)==1):
                req = req[1:-1]
                req = req.split(' ')
                return phrase(req,worddict_station)
            elif(querytype(req)==2):
                return prefix_query(req,tree,worddict_station,flag)
            elif(querytype(req)==3):
                return suffix_query(req,permuterm,worddict_station,flag)
            elif(querytype(req)==4):
                return mix_query(req,permuterm,tree,worddict_station,flag)

        elif querytype(inp) == 8:       #show query
            q = inp.split('=')
            req = q[1]
            if(querytype(req)==0):
                req = req.split(' ')
                req = preprocess_input(req)
                return search_postings(req,worddict_show,flag)
            elif(querytype(req)==1):
                req = req[1:-1]
                req = req.split(' ')
                return phrase(req,worddict_show)
            elif(querytype(req)==2):
                return prefix_query(req,tree,worddict_show,flag)
            elif(querytype(req)==3):
                return suffix_query(req,permuterm,worddict_show,flag)
            elif(querytype(req)==4):
                return mix_query(req,permuterm,tree,worddict_show,flag)

        elif querytype(inp) == 9:       #iashowid query
            q = inp.split('=')
            req = q[1]
            if(querytype(req)==0):
                req = req.split(' ')
                req = preprocess_input(req)
                return search_postings(req,worddict_iashowid,flag)
            elif(querytype(req)==1):
                req = req[1:-1]
                req = req.split(' ')
                return phrase(req,worddict_iashowid)
            elif(querytype(req)==2):
                return prefix_query(req,tree,worddict_iashowid,flag)
            elif(querytype(req)==3):
                return suffix_query(req,permuterm,worddict_iashowid,flag)
            elif(querytype(req)==4):
                return mix_query(req,permuterm,tree,worddict_iashowid,flag)

        elif querytype(inp) == 10:       #iapreviewid query
            q = inp.split('=')
            req = q[1]
            if(querytype(req)==0):
                req = req.split(' ')
                req = preprocess_input(req)
                return search_postings(req,worddict_iapreviewid,flag)
            elif(querytype(req)==1):
                req = req[1:-1]
                req = req.split(' ')
                return phrase(req,worddict_iapreviewid)
            elif(querytype(req)==2):
                return prefix_query(req,tree,worddict_iapreviewid,flag)
            elif(querytype(req)==3):
                return suffix_query(req,permuterm,worddict_iapreviewid,flag)
            elif(querytype(req)==4):
                return mix_query(req,permuterm,tree,worddict_iapreviewid,flag)

        elif querytype(inp) == 11:
            #snippet query
            flag = 2
            q = inp.split('=')
            req = q[1]
            if(querytype(req)==0):
                req = req.split(' ')
                req = preprocess_input(req) #Need to pass a variable to figure out if length normalization is possible
                return search_postings(req,worddict_snippet,flag)
            elif(querytype(req)==1):
                req = req[1:-1]
                req = req.split(' ')
                return phrase(req,worddict_snippet)
            elif(querytype(req)==2):
                return prefix_query(req,tree,worddict_snippet,flag)
            elif(querytype(req)==3):
                return suffix_query(req,permuterm,worddict_snippet,flag)
            elif(querytype(req)==4):
                return mix_query(req,permuterm,tree,worddict_snippet,flag)

        elif querytype(inp) == -1:
            print("Invalid input! Please try again")
            return {}


#---------------------------------- metrics stuff ------------------------
def get_prec_rec(our_res, elastic_res):
    #print("------------Main elastic res----------")
    #print(elastic_res)
    #print("------------ our res----------")
    pos = 0
    
    l = min(len(our_res), len(elastic_res))
    our_res = our_res[:l]
    elastic_res = elastic_res[:l]
    #print("Inside the compare function:", len(our_res), len(elastic_res))
    for i in our_res:
        d = json.loads(i)
        if 'tf-idf' in d.keys():
            removed_value = d.pop('tf-idf') 
        if (d in elastic_res):
            #print("!!!!!!!!!!!!!!!! MATCH !!!!!!!!!!!!!!!!!!!!!!!!!!!!", pos)
            #print(d)
            pos += 1
    
    prec = pos/len(elastic_res)
    recall = pos/len(our_res) 
    return prec, recall

def perform_elastic_search(queries):
    index = 'newsdb'
    es = Elasticsearch()
    all_results = []
    all_time = []
    
    for q in range(len(queries)):
        res = es.search(index=index,body=queries[q], size = 100)
        s = es.nodes.stats()
        for i in s['nodes']:
            print("Node :", i)
            print("Fetch time (in ms):", s['nodes'][i]['indices']['search']['fetch_time_in_millis'])
            print("Query time (in ms):", s['nodes'][i]['indices']['search']['query_time_in_millis'])
        
        q_result = []
        
        for x in res['hits']['hits']:
            obj = x['_source']
            q_result.append(obj)

        all_results.append(q_result)
        all_time.append(res['took'])

    return all_results, all_time

def perform_our_search(queries, worddict_all,worddict_url,worddict_matchdatetime,worddict_station,worddict_show,worddict_iashowid,worddict_iapreviewthumb,worddict_snippet,tree,permuterm):
    all_results = []
    all_time = []

    for q in range(len(queries)):
        begin = time.time()
        r = input_function(queries[q],worddict_all,worddict_url,worddict_matchdatetime,worddict_station,worddict_show,worddict_iashowid,worddict_iapreviewthumb,worddict_snippet,tree,permuterm)
        end = time.time()

        q_result = []
        if (len(r) != 0): 
            for x in r['row']:
                q_result.append(x)
    
        all_results.append(q_result)
        all_time.append(round((end - begin), 3)*1000)

    return all_results, all_time

def print_our_res(res):
    for i in range(len(res)):
        print("\n---- Doc ", i + 1, " ----\n")
        d = json.loads(res[i])
        for k,v in d.items():
            if (k != 'tf-idf'):
               print(k, "\t:", v, "\n")

def print_elastic_res(res):
    for i in range(len(res)):
        print("\n---- Doc ", i + 1, " ----\n")
        for k,v in res[i].items():
            print(k, "\t:", v, "\n")
        
def print_both_res(our_res, e_res):
    for i in range(min(len(our_res), len(e_res))):
        print("\n---- Our Doc ", i + 1, " ----\n")
        d = json.loads(our_res[i])
        for k,v in d.items():
            print(k, "\t:", v, "\n")
        print("\n---- Elastic search Doc ", i + 1, " ----\n")
        for k,v in e_res[i].items():
            print(k, "\t:", v, "\n")

def print_performance(query, our_results, our_time, elasic_results, elastic_time, num_q):
    prec = []
    rec = []
    f1_scores = []
    for i in range(num_q):
        print("\n============================ Query -", query[i], " ========================= ")
        if (len(our_results[i]) != 0):
            print("\n------------------------ Our search results -------------------")
            print_our_res(our_results[i][:5])
            print("\n---------------------- Elastic search results ----------------")
            print_elastic_res(elastic_results[i][:5])
            #print("\n---------------------- Both search results ----------------")
            #print_both_res(our_results[i], elastic_results[i])
            
            p, r = get_prec_rec(our_results[i], elastic_results[i])
            try:
                f1 = round((2*p*r)/(p + r), 3)
                f1_scores.append(f1)
            except:
                f1 = '-'
                #print("Prec and recall both are zero, so no F1")
            prec.append(p)
            rec.append(r)
            
            print("\nPrecision:", p, "\nRecall:", r, "\nF1:", f1, "\nTime taken by our search:", our_time[i], "\nTime taken by elastic search:", elastic_time[i])
            print()
        else:
            print("No results from our search")

    print("\n================ Averages ========================")
    print("\t   \tOur Search\tElastic Search")
    print(" Time taken\t", sum(our_time)/num_q, "\t\t", sum(elastic_time)/num_q)
    print(" Precision \t", sum(prec)/num_q)
    print(" Recall    \t", sum(rec)/num_q)
    print(" F1 Score  \t", sum(f1_scores)/num_q)

if __name__ == "__main__":
    
    worddict_all = read_from_disk("all.txt")
    worddict_url = read_from_disk("url.txt")
    worddict_matchdatetime = read_from_disk("matchdatetime.txt")
    worddict_station = read_from_disk("station.txt")
    worddict_show = read_from_disk("show.txt")
    worddict_iashowid = read_from_disk("iashowid.txt")
    worddict_iapreviewthumb = read_from_disk("iapreviewthumb.txt")
    worddict_snippet = read_from_disk("snippet.txt")
    tree = read_from_disk("twothreetreeall.txt")
    permuterm = read_from_disk("permutermall.txt")
    lengthdict = read_from_disk("length.txt")

    
    elastic_queries = [
    {
        "query": {
            "query_string": {
                "query":"president trump does not endorse climate change trump believes climate change is not real"
            }
        }
    },

    {
        "query": {
            "query_string": {
                "query":"european agenda"
            }
        }
    },

    {
        "query": {
            "query_string": {
                "query":"moz*"
            }
        }
    },


    {
         "query": {
         "match": { "Show": "Fox Friends" }}
    },

    ]

    '''
    elastic_queries = [
    {
         "query": {
         "match": { "Snippet": "mon*" }}
    },

    {
        "query": {
        "match": {
                "IAPreviewThumb":"https://archive.org/download/CNNW_20130401_230000_Erin_Burnett_OutFront/CNNW_20130401_230000_Erin_Burnett_OutFront.thumbs/CNNW_20130401_230000_Erin_Burnett_OutFront_000645.jpg"
        }
        }
    },

    {
         "query": {
         "match": { "IAShowID": "CNNW_20130401_230000_Erin_Burnett_OutFront" }}
    },

    {
         "query": {
         "match": { "Station": "CNN" }}
    },
    ]
    '''

    our_queries = ["president trump does not endorse climate change trump believes climate change is not real", "european agenda", "moz*", "Show=Fox Friends"]
    #our_queries = ["Snippet=will talk to scientists about climate change","IAPreviewThumb=https://archive.org/download/CNNW_20130401_230000_Erin_Burnett_OutFront/CNNW_20130401_230000_Erin_Burnett_OutFront.thumbs/CNNW_20130401_230000_Erin_Burnett_OutFront_000645.jpg", "IAShowID=CNNW_20130401_230000_Erin_Burnett_OutFront","Station=CNN"]

    elastic_results, elastic_time = perform_elastic_search(elastic_queries)
    our_results, our_time = perform_our_search(our_queries, worddict_all,worddict_url,worddict_matchdatetime,worddict_station,worddict_show,worddict_iashowid,worddict_iapreviewthumb,worddict_snippet,tree,permuterm)
    
    num_q = len(our_queries)
    print_performance(our_queries, our_results, our_time,elastic_results,elastic_time, num_q)
    
