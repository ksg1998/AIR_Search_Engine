# AIR_Search_Engine

## Software Requirements:

Important Python Packages needed:

pickle
spellchecker
stopwords
word_tokenize
PorterStemmer
WordNetLemmatizer
ElasticSearch


 1) Pre-Processing the input data and creating inverted index and saving to disk

Need to make inverted index for each column and all columns combined.

      python3 CreatingDictionary.py


 2) Creating the Permuterm index for the input data using the inverted index file obtained in the previous step.

      python CreatePermuterm.py


 3) Creating a two-three tree used for implementing wildcard queries. Preprocessed data is used as the input.

      python3 CreateTwoThree.py


 4) Running Our Search:

      python3 demo.py


 5) Comparing Our Search with Elastic:

      python3 main.py


