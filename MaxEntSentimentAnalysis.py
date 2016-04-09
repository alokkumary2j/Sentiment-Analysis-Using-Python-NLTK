
# coding: utf-8

# In[1]:

import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import MaxentClassifier
from nltk.corpus import movie_reviews
from nltk.metrics import scores
from nltk import precision
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# In[36]:

def evaluate_classifier(featx,collocationFunc):
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')
 
    negfeats = [(featx(movie_reviews.words(fileids=[f]),collocationFunc), 'neg') for f in negids]
    posfeats = [(featx(movie_reviews.words(fileids=[f]),collocationFunc), 'pos') for f in posids]

    lenNegFeats=min(len(negfeats),400)
    lenPosFeats=min(len(posfeats),400)
#    lenNegFeats=len(negfeats)
#    lenPosFeats=len(posfeats)
    negcutoff = int(lenNegFeats*3/4)
    poscutoff = int(lenPosFeats*3/4)
 
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:lenNegFeats] + posfeats[poscutoff:lenPosFeats]
 
    classifier = MaxentClassifier.train(trainfeats,algorithm='IIS',max_iter=3)
    print(classifier)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    print(classifier)
    for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
    evaluationMetrics={}
    classifier.show_most_informative_features()
    evaluationMetrics['model']=classifier
    evaluationMetrics['trainingData']=trainfeats
    evaluationMetrics['accuracy']=nltk.classify.util.accuracy(classifier, testfeats)
    evaluationMetrics['posPrec']=nltk.precision(refsets['pos'], testsets['pos'])
    evaluationMetrics['posRecall']=nltk.recall(refsets['pos'], testsets['pos'])
    evaluationMetrics['posF_Score']=nltk.f_measure(refsets['pos'], testsets['pos'])
    evaluationMetrics['negPrec']=nltk.precision(refsets['neg'], testsets['neg'])
    evaluationMetrics['negRecall']=nltk.recall(refsets['neg'], testsets['neg'])
    evaluationMetrics['negF_Score']=nltk.f_measure(refsets['neg'], testsets['neg'])
    return evaluationMetrics


# In[37]:

all_words = nltk.FreqDist(word for word in movie_reviews.words())
#type(all_words),type(all_words.keys())
dict_Keys=all_words.keys()
top_words=all_words.most_common(8000)
top_words=set(word[0] for word in top_words)
#{word[0]:word[1] for word in top_words)


# In[38]:

from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
evaluations=[]
def stopword_filtered_word_feats(words,collocator):
    return dict([(word, True) for word in words if word not in stopset if word in top_words])
evaluations.append(evaluate_classifier(stopword_filtered_word_feats,None)) 


# In[39]:

def returnUniqWC(evaluations):
    totUniqWords=[]
    for i in range(len(evaluations)):
        itemSet=set(evaluations[i][0].keys())
        totUniqWords.extend(list(itemSet))
        totUniqWords=set(totUniqWords)
        totUniqWords=list(totUniqWords)
    return len(totUniqWords)
maxEntModel=evaluations[0]["model"]
maxEntModel.ALGORITHMS#['GIS', 'IIS', 'MEGAM', 'TADM']
maxEntModel#<ConditionalExponentialClassifier: 2 labels, 6092 features>
len(evaluations[0]["trainingData"])#36 Entries (18 pos +18 neg Movie reviews)
posPart=evaluations[0]["trainingData"][0:18]
print(len(posPart),type(posPart))
negPart=evaluations[0]["trainingData"][18:36]
posWC=returnUniqWC(posPart)
negWC=returnUniqWC(negPart)
posWC,negWC,posWC+negWC#2959,3133,6092


# In[40]:

maxEntModel.classify_many([{"speech":True},{"speech":False},{"simple":True},{"unseenword":True},
                           {"killed":True},{"generally":True}])


# In[41]:

maxEntModel.explain({"speech":True})
print("*********************************************************************")
maxEntModel.explain({"killed":True})
print("*********************************************************************")
maxEntModel.explain({"generally":True})


# In[74]:

probDist1=maxEntModel.prob_classify({"speech":True})
print(probDist1.SUM_TO_ONE,probDist1.generate())
probDist2=maxEntModel.prob_classify({"generally":True})
probDist2.SUM_TO_ONE,probDist2.generate()


# In[10]:

#Bigram Collocations- Handle Cases like “not good”, here B-O-W Approach will Fail
def bigram_word_feats(words, score_fn, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

evaluations.append(evaluate_classifier(bigram_word_feats,BigramAssocMeasures.chi_sq))#Works best for this Data
#evaluations.append(evaluate_classifier(bigram_word_feats,BigramAssocMeasures.jaccard))
#evaluations.append(evaluate_classifier(bigram_word_feats,BigramAssocMeasures.likelihood_ratio))


# In[3]:

from nltk.collocations import *
from nltk.probability import FreqDist
from nltk.probability import ConditionalFreqDist
word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()

testNegWords = movie_reviews.words(categories=['pos'])
testPosWords = movie_reviews.words(categories=['neg'])

for word in testNegWords:
    word_fd[word.lower()]+=1
    label_word_fd['neg'][word.lower()]+=1
for word in testPosWords:
    word_fd[word.lower()]+=1
    label_word_fd['pos'][word.lower()]+=1
print(word_fd.N(),word_fd.B(),word_fd.most_common(20))
print(label_word_fd.N(),label_word_fd.conditions(),label_word_fd.items())
print(label_word_fd['pos'].N(),label_word_fd['neg'].N())


# In[ ]:

# n_ii = label_word_fd[label][word]
# n_ix = word_fd[word]
# n_xi = label_word_fd[label].N()
# n_xx = label_word_fd.N()
#         w1    ~w1
#      ------ ------
#  w2 | n_ii | n_oi | = n_xi
#      ------ ------
# ~w2 | n_io | n_oo |
#     ------ ------
#      =n_ix         TOTAL = n_xx
# A number of measures are available to score collocations or other associations. The arguments to measure 
# functions are marginals of a contingency table, in the bigram case (n_ii, (n_ix, n_xi), n_xx):
# n_ii = label_word_fd[label][word]
# n_ix = word_fd[word]
# n_xi = label_word_fd[label].N()
# n_xx = label_word_fd.N()
# Chi-Sq Contingency Table : Relating Word w1 with "pos" classification 
#         w1    ~w1
#      ------ ------
# +ve | n_ii | n_oi | = n_xi
#      ------ ------
# -ve | n_io | n_oo |
#     ------ ------
#      =n_ix         TOTAL = n_xx
# n_ix : Total Freq of word w1, n_xi: pos_word_count 
pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count
 
word_scores = {}

#print(word_fd.items())
for word, freq in word_fd.items():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],(freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],(freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score
import operator
best1 = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)[:10000]
bestwords = set([w for w, s in best1])
 
def best_word_feats(words,biGramMeasure):
    return dict([(word, True) for word in words if word in bestwords])
 
evaluations.append(evaluate_classifier(best_word_feats,BigramAssocMeasures.chi_sq))
 
def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words,score_fn))
    return d
#evaluations.append(evaluate_classifier(best_bigram_word_feats,BigramAssocMeasures.chi_sq))


# In[ ]:

for modelEvalMetrics in evaluations:
    print(modelEvalMetrics)

