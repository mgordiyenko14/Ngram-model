import sys
from collections import defaultdict
import math
import random
import os
import os.path


# NLTK function to generate ngrams
# import nltk
# from nltk.util import ngrams

"""
COMS W4705 - Natural Language Processing - Fall 2023 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

#returns the set aof all wordsthat appear in the corpus more than once
def get_lexicon(corpus): 
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    # TEST THIS MORE, somthig with printing START
    if n < 1:
        raise ValueError("The value of n should be greater than or equal to 1.")
    padded_sequence = ['START'] * (n - 1) + sequence + ['STOP']
    ngrams_list = [tuple(padded_sequence[i:i+n]) for i in range(len(padded_sequence) - n + 1)]
    return ngrams_list

class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # self.wordcount = sum(len(word) for sentence in generator for word in sentence)
    
        # Now iterate through the corpus again and count ngrams
        self.numberOfSentences = 0
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        self.total_tokens = self.get_wordcount(corpus_reader(corpusfile))

    def get_wordcount(self, corpus):
        count = 0
        for sentense in corpus:
            count += len(sentense)
        return count


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        # itetrate through the corpus
        #if a word appears more than once, add it to the dictionary and increment the counter
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        # self.unigramcounts = defaultdict(int)
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        #Your code here
        
        for sentence in corpus:#works
            self.numberOfSentences += 1
            sentence = [word if word in self.lexicon else 'UNK' for word in sentence]

            for word in get_ngrams(sentence,1):
                self.unigramcounts[word] = self.unigramcounts.get(word, 0) + 1

            bigrams = get_ngrams(sentence, 2)
            for bigram in bigrams:
                self.bigramcounts[bigram] = self.bigramcounts.get(bigram, 0) + 1

            trigrams = get_ngrams(sentence, 3)
            for trigram in trigrams:
                self.trigramcounts[trigram] = self.trigramcounts.get(trigram, 0) + 1


    def raw_trigram_probability(self,trigram):#works, do START START
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        trigram = self.unknown(trigram) 
        bigram = self.unknown(tuple(trigram[0:2])) 

        # bigram = trigram[:2]
        #check for START START
        if trigram not in self.trigramcounts:
            return 1/len(self.lexicon) #TA said ok
        if(bigram == (('START', 'START'))):
            return self.trigramcounts.get(trigram)/self.numberOfSentences
        if bigram not in self.bigramcounts:
            return 1/len(self.lexicon)
        else:
            res = self.trigramcounts.get(trigram)/self.bigramcounts.get(bigram)#idk
        return res
    

    def raw_bigram_probability(self, bigram): #works
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        unigram = []
        unigram.append(bigram[0])

        unigram = self.unknown(tuple(unigram)) 
        bigram = self.unknown(bigram) 
        
        if not bigram in self.bigramcounts:
            return 0.0
        if(unigram == ('START',)):
            return self.bigramcounts[bigram]/self.numberOfSentences
        else: res = float(self.bigramcounts[bigram]/self.unigramcounts.get(unigram))
        return res
    
    def raw_unigram_probability(self, unigram):# works
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        # print(len(self.lexicon))
        unigram = self.unknown(unigram) 
        if unigram == "START":
            return 0.0
        if unigram not in self.unigramcounts:
            return 0.0
        return self.unigramcounts.get(unigram)/self.total_tokens

    def unknown(self, joppa):# if not in lexicon - replaces with UNK
        return tuple("UNK" if word not in self.lexicon else word for word in joppa)


    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        prob_trigram = self.raw_trigram_probability(trigram)
        prob_bigram = self.raw_bigram_probability(trigram[1:3])
        prob_unigram = self.raw_unigram_probability(trigram[2:4])

        smoothed_prob = lambda1 * prob_trigram + lambda2 * prob_bigram + lambda3 * prob_unigram
        return smoothed_prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        # given a sentence, generaqte trigrams
        # for every trigram, fnd its smoothed probability, take a log of that
        # sum all the probabilities toegther to obtain the prob of a sentence
        trigram_list = get_ngrams(sentence, 3)
        log_prob = float(0)
        for trigram in trigram_list:
            log_prob +=math.log2(self.smoothed_trigram_probability(trigram))

        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        # For every sentence in the corpus, calcculate its log probability, (do not include the word start)
        # summ all of those probabilities
        # divide by the total number of words
        prob = 0.0
        words = 0
        for sentence in corpus:
            prob += self.sentence_logprob(sentence)
            words += len(sentence)

        # l = (float)(prob/words)
        perplexity = math.pow(2, -(float)(prob/words))

        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       

        for f in os.listdir(testdir1):
            pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if(pp_1<pp_2):
                correct +=1
            total +=1
    
        for f in os.listdir(testdir2):
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if(pp_2<pp_1):
                correct +=1
            total +=1
        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt', 'hw1_data/ets_toefl_data/train_low.txt', 'hw1_data/ets_toefl_data/test_high', 'hw1_data/ets_toefl_data/test_low')
    # print('Accuracy: ', acc)

