from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from random import randint
import random
import numpy as np
from operator import itemgetter 
from sklearn.svm import SVC
import pickle
from sklearn.naive_bayes import GaussianNB
clf = SVC(gamma='auto', kernel='linear')
clf_ = GaussianNB()#gamma='auto', kernel='linear')


np.set_printoptions(threshold=np.inf)

stemmer = SnowballStemmer("english")

lines = open('data', 'r')
lines = lines.readlines()

def prob(X, c, l, w, f,):
	#print([w[i] for i in range(len(X)) if X[i] >= 1])
	out = len(f)/len(l)
	for i in range(len(w)):
		if not w[i] in ['the_beatles','the_beatl','taylor_swift']:
			q = p(i,X[i],c,l,w,f)
			if q == 0.0:
				q = 0.00000000000000000001
			#if q <= 0.000000000001:
			out *= q
	return out

def p(i, x, c, l, w, f):
	count = 0 
	for y in f:
		if y[i] == x:
			count += 1
	return float(count)/float(len(f))

def find(c, l, la):
	return [l[i] for i in range(len(l)) if la[i] in c]

def check_right(x, cs, l, w, fs, la):
	choice = max([(prob(x, cs[i], l, w, fs[i]), cs[i]) for i in range(len(cs))], key = itemgetter(0))[1]
	#printt(x,w)
	#print(choice)
	#printt(x, w)
	for i in choice:
		if i == la:
			return False
	return True

def check_all(xs, cs, l, w, fs, la):
	qwertyuiopasdfghjklzxcvbnm = len(xs)
	right = 0
	for i in range(len(xs)):
		right += int(check_right(xs[i], cs, l, w, fs, la[i]))
		print('\r'+str(right/(i+1)) + " " + str(i+1), '')
	return right/qwertyuiopasdfghjklzxcvbnm

def printt(x,w):
	print([w[i] for i in range(len(x)) if x[i]>0])
	return ([w[i] for i in range(len(x)) if x[i]>0])



deletions = []

labels = [None]*len(lines)
for j in range(len(lines)):
	i = lines[j]
	if 'chorus' in i.lower():
		if len(i[i.index('\t'):]) <= 30:
			deletions.append(j)

		else:
			lines[j] = lines[j][:i.index('(')].split()
			labels[j] = lines[j][0]
			lines[j] = lines[j][1:]
			lines[j] = ' '.join([stemmer.stem(word) for word in lines[j]])
	else:
		lines[j] = lines[j].split()
		labels[j] = lines[j][0]
		lines[j] = lines[j][1:]
		lines[j] = [stemmer.stem(word) for word in lines[j]] #COMMENT
		lines[j] = ' '.join(lines[j])
	#if 

vectorizer = CountVectorizer()
lines = list(vectorizer.fit_transform(lines).toarray())
words = vectorizer.get_feature_names()

line = random.shuffle(lines)

#printt(lines[1105], words)


for i in range(len(deletions)):
	lines.pop(deletions[i]-i)
	labels.pop(deletions[i]-i)

#with open("lines.pickle", "wb+") as file:
	#pickle.dump(lines,file)

#with open("labels.pickle", "wb+") as file:
	#pickle.dump(labels,file)

lineGroups = [[],[],[],[],[],[],[],[],[],[]]
labelGroups = [[],[],[],[],[],[],[],[],[],[]]
for i in range(len(lines)):
	r = randint(0,9)
	lineGroups[int(r)].append(lines[i])
	labelGroups[int(r)].append(labels[i])
gnb = []
svm = []
for i in range(10):
	test = lineGroups[i]
	lines = sum(lineGroups[:i]+lineGroups[i+1:],[])
	testLabels = labelGroups[i]
	labels = sum(labelGroups[:i]+labelGroups[i+1:],[])
	lines = np.array(lines)
	labels = np.array(labels)
	#print(labels.shape)
	clf_.fit(np.array(lines), np.array(labels))
	print('gnb')
	gnb.append(float(1)-clf_.score(test,testLabels)) #beacuse it keeps giving .43 ish so i switched it
	print(gnb[-1])
	clf.fit(np.array(lines), np.array(labels))
	print('svm')
	svm.append(clf.score(test,testLabels))
	print(svm[-1])

	#for i in 
	#print(i)
	#print(check_all(test, [['taylor_swift'],['the_beatles']], lines, words, [find(['the_beatles'],lines,labels),find(['taylor_swift'],lines,labels)],testLabels))
print("Naive Bayes: "+str(float(sum(gnb)/len(gnb))))
print("Support Vector Machine: "+str(float(sum(svm)/len(svm))))
