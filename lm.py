#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from collections import defaultdict
from math import log
import sys

eos = "END_OF_SENTENCE"

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob(eos, sentence)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass


class Unigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.trigrams = dict()
        self.tri_context = dict()
        self.lbackoff = log(backoff, 2)

    def inc_word(self, w):
        if w in self.trigrams:
            self.trigrams[w] += 1.0
        else:
            self.trigrams[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word(eos)

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.trigrams:
            tot += self.trigrams[word]
        ltot = log(tot, 2)
        for word in self.trigrams:
            self.trigrams[word] = log(self.trigrams[word], 2) - ltot

    def cond_logprob(self, word, previous):
        if word in self.trigrams:
            return self.trigrams[word]
        else:
            return self.lbackoff

    def vocab(self):
        return self.trigrams.keys()


class Trigram(LangModel):
    def __init__(self, backoff = 0.000001, l1 = .9, l2 = .1):
        self.trigrams = dict()
        self.tri_context = defaultdict(int)
        self.bigrams = dict()
        self.bi_context = defaultdict(int)
        self.lbackoff = log(backoff, 2)
        self.vocabulary = set([eos])
        self.l1 = l1
        self.l2 = l2


    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob(eos, sentence)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence):
        sentence = ["START_OF_SENTENCE", "START_OF_SENTENCE"] + sentence + [eos]
        for i in range(2, len(sentence)):
            self.vocabulary.add(sentence[i])
            self.increment((sentence[i-2], sentence[i-1], sentence[i]))
        if len(sentence) > 1:
            self.increment((sentence[-2], sentence[-1], eos))

    def increment(self, t):
        if t in self.trigrams:
            self.trigrams[t] += 1.0
        else:
            self.trigrams[t] = 1.0
        if (t[0], t[1]) in self.tri_context:
            self.tri_context[(t[0], t[1])] += 1.0
        else:
            self.tri_context[(t[0], t[1])] = 1.0        

        b = (t[1], t[2])
        if b in self.bigrams:
            self.bigrams[b] += 1.0
        else:
            self.bigrams[b] = 1.0
        if b[0] in self.bi_context:
            self.bi_context[b[0]] += 1.0
        else:
            self.bi_context[b[0]] = 1.0      


    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.trigrams:
            tot += self.trigrams[word]
        ltot = log(tot, 2)
        for word in self.trigrams:
            self.trigrams[word] = log(self.trigrams[word], 2) - ltot
        tot = 0.0
        for con in self.tri_context:
            tot += self.tri_context[con]
        ltot = log(tot, 2)
        for con in self.tri_context:
            self.tri_context[con] = log(self.tri_context[con], 2) - ltot
        
        tot = 0.0
        for word in self.bigrams:
            tot += self.bigrams[word]
        ltot = log(tot, 2)
        for word in self.bigrams:
            self.bigrams[word] = log(self.bigrams[word], 2) - ltot
        tot = 0.0
        for con in self.bi_context:
            tot += self.bi_context[con]
        ltot = log(tot, 2)
        for con in self.bi_context:
            self.bi_context[con] = log(self.bi_context[con], 2) - ltot


    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous):
        if len(previous) == 0:  # first word in sentence
            prev_word = "START_OF_SENTENCE"
            prev_prev_word = "START_OF_SENTENCE"
        elif len(previous) == 1:
            prev_word = previous[-1]
            prev_prev_word = "START_OF_SENTENCE"
        else:
            prev_word = previous[-1]
            prev_prev_word = previous[-2]


        tot = len(self.vocabulary)   
        tri = (prev_prev_word, prev_word, word) 
        bi = (prev_word, word)
        ret = 0
        if tri in self.trigrams:
            ret += self.l1 * (self.trigrams[tri])/(self.tri_context[(tri[0], tri[1])]+tot)
        if bi in self.bigrams:
            ret += self.l2 * (self.bigrams[bi])/(self.bi_context[bi[0]]+tot)
        else: 
            ret += self.lbackoff
        return ret

    # required, the list of words the language model suports (including EOS)
    def vocab(self):
        return list(self.vocabulary)

