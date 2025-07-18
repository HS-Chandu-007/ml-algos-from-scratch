import numpy as np
import re
from collections import defaultdict

training_data = [
    ("buy now", "spam"),
    ("limited offer", "spam"),
    ("hello friend", "ham"),
    ("how are you", "ham")
]

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

class NaiveBayesClassifier:
    def __init__(self):
        self.classwordscount = defaultdict(lambda: defaultdict(int))
        self.classcount = defaultdict(int)
        self.vocab = set()
        self.totalwordsperclass = defaultdict(int)
        self.classprobs = {}
        
    def train(self, data):
        for text, label in data:
            self.classcount[label] +=1
            words = tokenize(text)
            for words in words:
                self.classwordscount[label][words] += 1
                self.totalwordsperclass[label] += 1
                self.vocab.add(words)

        totaldocs = sum(self.classcount.values())
        self.classprobs = {
            label: count/ totaldocs for label, count in self.classcount.items()
        }

    
    def predict(self, text):
        words = tokenize(text)
        scores = {}
    
        for label in self.classcount:
            logprob = np.log(self.classprobs[label])
            totalwords = self.totalwordsperclass[label]
            vocadsize = len(self.vocab)

            for word in words:
                wordcount = self.classwordscount[label][word]

                prob = (wordcount + 1) + (totalwords + vocadsize)
                logprob += np.log(prob)

                scores[label] =logprob

        return max(scores, key=scores.get) 


# Training data
training_data = [
    ("buy now", "spam"),
    ("limited offer", "spam"),
    ("hello friend", "ham"),
    ("how are you", "ham")
]

model = NaiveBayesClassifier()
model.train(training_data)

# Test
test_messages = [
    "buy one now",
    "hello how are you",
    "limited time friend"
]

for msg in test_messages:
    prediction = model.predict(msg)
    print(f"Message: '{msg}' => Prediction: {prediction}")

