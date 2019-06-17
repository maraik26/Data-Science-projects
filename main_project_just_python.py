#from helper import *
#python -m pip install <name of packages/library>
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim.models.ldamodel import LdaModel as Lda
from gensim import corpora
import string
import os
import codecs
import pyLDAvis
import pandas as pd
import pyLDAvis.gensim
import warnings
import matplotlib
from openpyxl import Workbook
matplotlib.use('Agg')
warnings.filterwarnings("ignore")
import gensim
import warnings
import pandas as pd
from pylab import figure
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# # add log for recording the model fitting data while training
lemma = WordNetLemmatizer()

from time import time
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                   filename='running.log',filemode='w')

comments=pd.read_csv("cleaning.csv") # after you create and save a file using SPARK in a different program


comments = comments.tokens_Stop[1:650000].tolist() # choose a number of cleaned comments to work on

# Function to remove stop words from sentences, punctuation & lemmatize words.
def clean(doc):
    exclude = set(string.punctuation)
    translate_table = dict((ord(char), None) for char in string.punctuation)
    no_punct = doc.lower().translate(translate_table)
    stop_free = " ".join([i for i in no_punct.split()])
    normalized = " ".join(lemma.lemmatize(word,'v') for word in stop_free.split())
    
    x = normalized.split()
    y = [s for s in x if len(s) > 2]
    return y

cleanPost = [clean(doc) for doc in comments]

print(len(cleanPost))

# Find the most frequent words and exclude NEUtral them. My bias!!! May be work more on that?
import itertools
flattened_cleanPost = list(itertools.chain(*cleanPost))

from collections import Counter
word_counts = Counter(flattened_cleanPost)
top_three = word_counts.most_common(200)
print(top_three)

most_fr = pd.DataFrame(top_three, columns=['words','count']).set_index('words')
most_fr.to_excel('most_frequent_650000_20t.xlsx')

#CHART MOST FREQUENT
fiz=plt.figure(figsize=(8,8))
plt.title('Most frequent words')
sns.barplot(x='count', y=most_fr[1:30].index, data=most_fr[1:30], label='Cities', palette='Reds_d')
plt.xlabel('count')
fiz.savefig('650000_20t_MF.png')
#plt.show()

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(cleanPost)
#dictionary.save('dictionary.dict')
print (dictionary)

#After printing the most frequent time back good right words of the dictionary, I found that few words which are mostly content neutral words are also present in the dictionary. 
# These words may lead to modeling of “word distribution”(topic) which is neutral and do not capture any theme or content. 
# I made a list of such words and filtered all such words.
stoplist = set('low even see less much disagree self bye obviously still that you more another mine instead never ask as so per will every instead never ask want leave bring give one tell say new try take may come get two three get would seem like want hey might may without also make want put etc actually else far definitely youll\' didnt\' isnt\' theres since able maybe without may suggestedsort never isredditmediadomain userreports far appreciate next think know need look please one null take dont dont\' want\' could able ask well best someone sure lot thank also anyone really something give years use make all ago people know many call include part find become '.split())
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)

#build a corpus and save it for a future
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above and save for a future use
doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleanPost]

#corpora.MmCorpus.serialize('corpus.mm', doc_term_matrix)
print (len(doc_term_matrix))

start = time()
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

#Creating the object for LDA model using gensim library & Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=20, id2word = dictionary, passes=20, iterations=500)
print ('used LDA: {:.2f}s'.format(time()-start))

#print(ldamodel.print_topics(num_topics=4, num_words=30))

ii = ldamodel.print_topics(num_topics=20, num_words=30)
df = pd.DataFrame(ii, columns=['id_topics', 'words']).set_index('id_topics')
#df = df.to_csv('topics.csv')
df = df.to_excel('650000_20t.xlsx')

#for topic in ldamodel.print_topics(num_topics=4, num_words=30):
#    print (topic[0]+1, " ", topic[1],"\n")

#ldamodel.save('topic_comments_lda.model')

#MAIN PLOT
viz = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)
pyLDAvis.save_html(viz, '650000_20t.html')

#ldamodel.save('TM_lda_1000_4t.model')


#FUGURES
fig = plt.figure(figsize=(15,30))
for i in range(20):
    df=pd.DataFrame(ldamodel.show_topic(i), columns=['term','prob']).set_index('term')
#     df=df.sort_values('prob')
    
    plt.subplot(10,2,i+1)
    plt.title('topic '+str(i+1))
    sns.barplot(x='prob', y=df.index, data=df, label='Cities', palette='GnBu_d')
    plt.xlabel('probability')
    
    
#plt.show()
fig.savefig('650000_20t.png')



