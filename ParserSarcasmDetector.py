import pandas as pd
from textblob import TextBlob
from sklearn.cross_validation import train_test_split
from textblob.en.parsers import PatternParser
import re
from textblob.taggers import PatternTagger
import ahocorasick
import esmre
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import esm
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.cross_validation import KFold
from tqdm import tqdm

import sys
stdin, stdout = sys.stdin, sys.stdout
reload(sys)
sys.stdin, sys.stdout = stdin, stdout
sys.setdefaultencoding('utf-8')


class ParserSarcasmDetector:
    def __init__(self,d):
        self.d = d
        self.positive_sentiments = esm.Index()
        self.negative_sentiments = esm.Index()
        self.positive_situations = esm.Index()
        self.negative_situations = esm.Index()
        self.s_positive_sentiments = set()
        self.s_negative_sentiments = set()
        self.s_positive_situations = set()
        self.s_negative_situations = set()
         
        
    def fit(self, data):
        self.pblga(data)
        
    def predict(self, data, d):
        self.d = d
        predictions = []
        for t in data:
            p = self.iws(t)
            if p is not None:
                predictions.append(p)
            else:
                ps = self.positive_sentiments.query(t.lower())
                ns = self.negative_situations.query(t.lower())
                if self.check_minimal_distance(ps, ns, t):
                    predictions.append(1)
                    continue
                ps = self.positive_situations.query(t.lower())
                ns = self.negative_sentiments.query(t.lower())
                if self.check_minimal_distance(ps, ns, t):
                    predictions.append(1)
                    continue
                predictions.append(0)
        return predictions
    

    def predict_simple(self, data):
        predictions = []
        for t in data:
            p = self.iws(t)
            if p is not None:
                predictions.append(p)
            else:
                ps = self.positive_sentiments.query(t.lower())
                ns = self.negative_situations.query(t.lower())
                if (len(ps) > 0) and (len(ns) > 0):
                    predictions.append(1)
                    continue
                ps = self.positive_situations.query(t.lower())
                ns = self.negative_sentiments.query(t.lower())
                if (len(ps) > 0) and (len(ns) > 0):
                    predictions.append(1)
                    continue
                predictions.append(0)
        return predictions
    

    
    def check_minimal_distance(self, ps, ns, st):
        for w1 in ps:
            for w2 in ns:
                m_w1 = re.search(r'b'+re.sub(r"[|]", r"",w1[1])+r'b', st.lower())
                m_w2 = re.search(r'b'+re.sub(r"[|]", r"",w2[1])+r'b', st.lower())
                if (m_w1 is not None) and (m_w2 is not None):
                    w = w1[1].count(" ") + w2[1].count(" ")+w1[1].count("n") + w2[1].count("n")
                    s1 = m_w1.start()
                    e1 = m_w1.end()-1
                    s2 = m_w2.start()
                    e2 = m_w2.end()-1 
                    if s1 < s2:
                        s = s1
                    else:
                        s = s2
                    if e1 > e2:
                        e = e1
                    else:
                        e = e2
                    sl = st[s:e+1]
                    w = sl.count(" ") + sl.count("n") - w
                    if w > 0 and w < self.d:
                        return True
        return False
                    
                

    def pblga(self, data):
        pf = set()
        sentiments = set()
        situations = set()
        for t in data:
            pf.add(TextBlob(t.lower()).parse())
        for tw in pf:
            t = tw.split()
            t = [item for sublist in t for item in sublist]
            phrases = self.get_phrase(t)
            for ind, t in enumerate(phrases):
                if (t[1] == "NP") or (t[1] == "ADJP"):
                    sentiments.add(t[0])
                if (t[1] == "NP") and  (ind < len(phrases) -1) and (phrases[ind+1][1] == "VP"):
                    sentiments.add(t[0] + " " + phrases[ind+1][0])
                if t[1] == "VP":
                    situations.add(t[0])
                if (ind < len(phrases) - 1) and (((t[1] == "ADVP") and  (phrases[ind+1][1] == "VP"))
                    or ((t[1] == "ADJP") and  (phrases[ind+1][1] == "VP")) or
                    ((t[1] == "VP") and  (phrases[ind+1][1] == "NP"))):
                    situations.add(t[0] + " " + phrases[ind+1][0])
                if (ind < len(phrases) - 2) and (((t[1] == "VP") and (phrases[ind+1][1] == "ADVP")
                    and (phrases[ind+2][1] == "ADJP")) or
                    ((t[1] == "VP") and (phrases[ind+1][1] == "ADJP")
                    and (phrases[ind+2][1] == "NP")) or
                    ((t[1] == "ADVP") and (phrases[ind+1][1] == "ADJP")
                    and (phrases[ind+2][1] == "NP"))):
                    situations.add(t[0] + " " + phrases[ind+1][0] + " " + phrases[ind+2][0])
        for p in sentiments:
            ss = self.get_sentiment_score(p)
            if ss > 0:
                if p not in self.s_positive_sentiments:
                    self.s_positive_sentiments.add(p)
                    self.positive_sentiments.enter(p)
            elif ss < 0:
                if p not in self.s_negative_sentiments:
                    self.s_negative_sentiments.add(p)
                    self.negative_sentiments.enter(p)
        for p in situations:
            ss = self.get_sentiment_score(p)
            if ss > 0:
                if p not in self.s_positive_situations:
                    self.s_positive_situations.add(p)
                    self.positive_situations.enter(p)
            elif ss < 0:
                if p not in self.s_negative_situations:
                    self.s_negative_situations.add(p)
                    self.negative_situations.enter(p)
        
        
    def fix(self):
        self.positive_sentiments.fix()
        self.positive_situations.fix()
        self.negative_sentiments.fix()
        self.negative_situations.fix()
    
    
    def get_phrase(self, text):
        np = re.compile('.*NP')
        vp = re.compile('.*VP')
        advp = re.compile('.*ADVP')
        adjp = re.compile('.*ADJP')
        pred = None
        cur = None
        phrases = []
        phrase = ""
        for t in text:
            if np.match(t[2]):
                cur = "NP"
            elif vp.match(t[2]):
                cur = "VP"
            elif advp.match(t[2]):
                cur = "ADVP"
            elif adjp.match(t[2]):
                cur = "ADJP"
            else:
                cur = None
            if pred != None:
                phrases.append((phrase, pred))
                phrase = ""
                pred = None
            if cur == pred:
                phrase = phrase + " " + t[0]
            elif pred == None:
                pred = cur
                phrase = t[0]
            else:
                phrases.append((phrase, pred))
                pred = cur
                phrase = t[0]
        if pred != None:
            phrases.append((phrase, pred))
        return phrases
    
    def get_sentiment_score(self, text):
        text = text.split()
        positive = 0
        negative = 0
        for word in text:
            s = []
            sn = TextBlob(word.lower()+" ").polarity
            if sn != 0:
                s.append(sn * 5)
            ss = senti_score(word.lower())
            if ss != 0:
                s.append(sn)
            polarity = np.mean(s or [0])
            if polarity > 0:
                positive += 1
            elif polarity < 0:
                negative += 1
        pr = positive * 1.0 / len(text)
        nr = negative * 1.0 / len(text)
        return pr - nr
    
    def iws(self, text):
        verbs = re.compile('VB*')
        nouns = re.compile('NN*')
        adjectives = re.compile('JJ*')
        adverbs = re.compile('RB*')
        blob = TextBlob(text, pos_tagger=PatternTagger())
        tags = blob.pos_tags
        for index, t in enumerate(tags):
            if t[1]=='UH':
                if (index < len(tags)-1) and (adjectives.match(tags[index+1][1]) or
                                              adverbs.match(tags[index+1][1])):
                    return 1
                for i, next_tag in enumerate(tags[index+1:]):
                    if (i < len(tags[index+1:]) - 1) and ((adverbs.match(next_tag[1]) 
                                                       and adjectives.match(tags[index+1+i+1][1]))or
                     (adjectives.match(next_tag[1]) and nouns.match(tags[index+1+i+1][1])) or
                     (adverbs.match(next_tag[1]) and verbs.match(tags[index+1+i+1][1])) ):
                        return 1
                return 0
        return None