## Search Engine
This Search engine has been built as part of the Fall 2020 offering of the Algorithms for Information Retrieval Course taught by Prof. Bhaskar Jyothi Das at PES University, Bangalore, India. <br>
The search engine has been built on an Environmental News NLP archive corpus consisting of 418 documents. 
The search engine supports the following types of queries:
<ul>
 <li>free text queries</li>
 <li>phrase queries</li>
 <li>wildcard queries</li>
 <li>column specific queries</li>
</ul>

The code has been implemented completely in Python. 
Largely, the functionalities implemented include:
<ul>
 <li>Creating the inverted index</li>
 <li>Retrieving relevant documents for the user query</li>
 <li>Calculating metrics inorder to quantify the performance</li>
</ul>
Elasticsearch is a search engine based on the Lucene library and it has been used as a benchmark for a comparison study.<br>

## Overview
<br>
Building the search engine
![Building the search engine](/Images/workflow1.PNG)
<br>
Handling user query
![Handling user query](/Images/workflow2.PNG)
## Software Requirements:
<ul>
 <li>Python3 - <a href="https://www.python.org/">Download & Install Python3</a>, and ensure that it is latest version to avoid any version clashes.</li>
 <li>Download and import the following python packages:
  <ul>
   <li>pickle</li>
 <li>spellchecker</li>
 <li>stopwords</li>
 <li>word_tokenize</li>
 <li>PorterStemmer</li>
 <li>WordNetLemmatizer</li>
 <li>ElasticSearch</li>
  </ul>
 </ul>
 
## Deploy the application


 1) Pre-Processing the input data and creating inverted index and saving to disk

Need to make inverted index for each column and all columns combined
```bash
  $ python3 CreatingDictionary.py
```

 2) Creating the Permuterm index for the input data using the inverted index file obtained in the previous step
```bash    
  $ python CreatePermuterm.py
```

 3) Creating a two-three tree used for implementing wildcard queries. Preprocessed data is used as the input
```bash
  $ python3 CreateTwoThree.py
```

 4) Running Our Search
```bash
  $ python3 demo.py
```

 5) Comparing Our Search with Elastic
```bash
  $ python3 main.py
```
## Overview
Free Text query
![Free Text query](/Images/FreeTextQuery.png)
Wild Card query
![Wild Card query](/Images/WildCardQuery.png)
## About the Team
<ul>
  <li><a href = "https://github.com/bharaniuk">Bharani Ujjaini Kempaiah</a></li>
  <li><a href = "https://github.com/BhavyaCharan">Bhavya Charan</a></li>
  <li><a href = "https://github.com/rubenjohn1999">Ruben John Mampilli</a></li>
  <li><a href = "https://github.com/ksg1998">Goutham KS</a></li>
 </ul>
