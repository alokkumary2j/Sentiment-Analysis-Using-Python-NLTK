
# coding: utf-8

# In[60]:

import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from nltk.corpus import movie_reviews
from nltk.metrics import scores
from nltk import precision
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier


# In[63]:

def evaluate_classifier(featx,collocationFunc):
    #negFiles = movie_reviews.fileids('neg')
    #posFiles = movie_reviews.fileids('pos')
    #negWordsList=[movie_reviews.words(fileids=[f]) for f in negFiles]
    #posWordsList=[movie_reviews.words(fileids=[f]) for f in posFiles]
    #negfeats = [(featx(negWords), 'neg') for negWords in negWordsList]
    #posfeats = [(featx(posWords), 'pos') for posWords in posWordsList]

    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')
 
    negfeats = [(featx(movie_reviews.words(fileids=[f]),collocationFunc), 'neg') for f in negids]
    posfeats = [(featx(movie_reviews.words(fileids=[f]),collocationFunc), 'pos') for f in posids]

    lenNegFeats=min(len(negfeats),24)
    lenPosFeats=min(len(posfeats),24)
    print("Sample DataSets : ",lenNegFeats,lenPosFeats)
    #lenNegFeats=len(negfeats)
    #lenPosFeats=len(posfeats)
    negcutoff = int(lenNegFeats*3/4)
    poscutoff = int(lenPosFeats*3/4)
 
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:lenNegFeats] + posfeats[poscutoff:lenPosFeats]
 
    classifier = SklearnClassifier(LinearSVC()).train(trainfeats)
    #classifier = SklearnClassifier(SVC()).train(trainfeats)
    #classifier = SklearnClassifier(BernoulliNB()).train(trainfeats)
    #classifier = SklearnClassifier(MultinomialNB()).train(trainfeats)
    #classifier = SklearnClassifier(GaussianNB()).train(trainfeats) #Doesn't make sense for Non-Numeric Values
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
 
    for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
    evaluationMetrics={}
    print(classifier)
    evaluationMetrics['svmModel']=classifier
    #evaluationMetrics['bernModel']=bernoulliModel
    evaluationMetrics['accuracy']=nltk.classify.util.accuracy(classifier, testfeats)
    evaluationMetrics['posPrec']=nltk.precision(refsets['pos'], testsets['pos'])
    evaluationMetrics['posRecall']=nltk.recall(refsets['pos'], testsets['pos'])
    evaluationMetrics['posF_Score']=nltk.f_measure(refsets['pos'], testsets['pos'])
    evaluationMetrics['negPrec']=nltk.precision(refsets['neg'], testsets['neg'])
    evaluationMetrics['negRecall']=nltk.recall(refsets['neg'], testsets['neg'])
    evaluationMetrics['negF_Score']=nltk.f_measure(refsets['neg'], testsets['neg'])
    return evaluationMetrics


# In[64]:

from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
evaluations=[] 
def stopword_filtered_word_feats(words,collocator):
    return dict([(word, True) for word in words if word not in stopset])
evaluations.append(evaluate_classifier(stopword_filtered_word_feats,None)) 


# In[65]:

evaluations[0]
#******************** RESULTS FOR SVC :24 Reviews*************************************
#{'accuracy': 0.583,'negF_Score': 0.705,'negPrec': 0.545,'negRecall': 1.0,'posF_Score': 0.285,'posPrec': 1.0,
#'posRecall': 0.166,'svmModel': <SklearnClassifier(SVC(C=1.0, cache_size=200,degree=3, gamma='auto', kernel='rbf',
   #max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False))>}
#******************** RESULTS FOR SVC :120 Reviews*************************************
#{'accuracy': 0.6,'negF_Score': 0.700,'negPrec': 0.56,'negRecall': 0.93,'posF_Score': 0.4,'posPrec': 0.8,
#'posRecall': 0.266,'svmModel': <SklearnClassifier(SVC(C=1.0, cache_size=200,degree=3, gamma='auto', kernel='rbf',
   #max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False))>}
#******************** RESULTS FOR SVC :1000 Reviews*************************************
#{'accuracy': 0.632,'negF_Score': 0.729,'negPrec': 0.576,'negRecall': 0.992,'posF_Score': 0.425,'posPrec': 0.971,
#'posRecall': 0.272,'svmModel': <SklearnClassifier(SVC(C=1.0, cache_size=200,degree=3, gamma='auto', kernel='rbf',
   #max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False))>}

#******************** RESULTS FOR LINEAR SVC :24 Reviews*************************************
#{'accuracy': 0.916,'negF_Score': 0.923,'negPrec': 0.857,'negRecall': 1.0,'posF_Score': 0.909,'posPrec': 1.0,
#'posRecall': 0.833,'svmModel': <SklearnClassifier(LinearSVC(C=1.0,intercept_scaling=1, loss='squared_hinge', 
    #max_iter=1000,multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,verbose=0))>}
#******************** RESULTS FOR LINEAR SVC :240 Reviews*************************************
#{'accuracy': 0.741,'negF_Score': 0.755,'negPrec': 0.716,'negRecall': 0.8,'posF_Score': 0.725,'posPrec': 0.773,
#'posRecall': 0.683,'svmModel': <SklearnClassifier(LinearSVC(C=1.0,intercept_scaling=1, loss='squared_hinge', 
    #max_iter=1000,multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,verbose=0))>}
#******************** RESULTS FOR LINEAR SVC :1000 Reviews*************************************
#{'accuracy': 0.872,'negF_Score': 0.874,'negPrec': 0.857,'negRecall': 0.892,'posF_Score': 0.869,'posPrec': 0.887,
#'posRecall': 0.852,'svmModel': <SklearnClassifier(LinearSVC(C=1.0,intercept_scaling=1, loss='squared_hinge', 
    #max_iter=1000,multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,verbose=0))>}

#******************** RESULTS FOR BernoulliNB :24 Reviews*************************************
#'accuracy': 0.75,'negF_Score': 0.8,'negPrec': 0.666, 'negRecall': 1.0,
#'posF_Score': 0.666,'posPrec': 1.0,'posRecall': 0.5
#******************** RESULTS FOR BernoulliNB :240 Reviews*************************************
#'accuracy': 0.733,'negF_Score': 0.768,'negPrec': 0.679, 'negRecall': 0.883,
#'posF_Score': 0.686,'posPrec': 0.833,'posRecall': 0.583
#******************** RESULTS FOR BernoulliNB :1000 Reviews*************************************
#'accuracy': 0.808,'negF_Score': 0.8285,'negPrec': 0.748, 'negRecall': 0.928,
#'posF_Score': 0.781,'posPrec': 0.905,'posRecall': 0.688

#******************** RESULTS FOR MultiNomialNB :24 Reviews*************************************
#'accuracy': 0.75,'negF_Score': 0.8,'negPrec': 0.666, 'negRecall': 1.0,
#'posF_Score': 0.666,'posPrec': 1.0,'posRecall': 0.5
#******************** RESULTS FOR MultiNomialNB :240 Reviews*************************************
#'accuracy': 0.791,'negF_Score': 0.8,'negPrec': 0.769, 'negRecall': 0.833,
#'posF_Score': 0.782,'posPrec': 0.818,'posRecall': 0.75
#******************** RESULTS FOR MultiNomialNB :1000 Reviews*************************************
#'accuracy': 0.826,'negF_Score': 0.832,'negPrec': 0.802, 'negRecall': 0.864,
#'posF_Score': 0.819,'posPrec': 0.852,'posRecall': 0.788


# In[51]:

svmModel=evaluations[0]['svmModel']
bernoulliModel=evaluations[0]['bernModel']
print(len(bernoulliModel.feature_count_[0]),len(bernoulliModel.feature_count_[1]),bernoulliModel.alpha)
print(bernoulliModel.classes_,len(bernoulliModel.coef_[0]),bernoulliModel.coef_[0][0])
bernoulliModel.class_count_#[18.0,18.0]
print(bernoulliModel.classes_)#[0,1]
bernoulliModel.get_params()#{'alpha': 1.0, 'binarize': 0.0, 'class_prior': None, 'fit_prior': True}
bernoulliModel.class_log_prior_#[ln(0.5)=-0.69314718, ln(.5)=-0.69314718]
bernoulliModel.
#BernoulliNB.get_params(svmModel)


# In[13]:

#Bigram Collocations- Handle Cases like “not good”, here B-O-W Approach will Fail
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
 
def bigram_word_feats(words, score_fn, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
 
evaluations.append(evaluate_classifier(bigram_word_feats,BigramAssocMeasures.chi_sq))#Works best for this Data
evaluations.append(evaluate_classifier(bigram_word_feats,BigramAssocMeasures.jaccard))
evaluations.append(evaluate_classifier(bigram_word_feats,BigramAssocMeasures.likelihood_ratio))


# In[ ]:

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
evaluations.append(evaluate_classifier(best_bigram_word_feats,BigramAssocMeasures.chi_sq))


# In[ ]:

for modelEvalMetrics in evaluations:
    print(modelEvalMetrics)

