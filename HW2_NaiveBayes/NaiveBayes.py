import sys
import os
import string
from nltk.stem import PorterStemmer 
import math

def ReadStopWordFile(stopFile):
    stopwords = []
    path = stopFile
    with open(path, 'r') as f:
        for line in f:
            line = line.translate(str.maketrans('', '', string.punctuation))
            for word in line.split():
                stopwords.append(word.lower())
    return stopwords

def ApplyMultinomialNaiveBayes(ClassCount,extractVocab, prior, spamVocabDict, hamVocabDict, vocabList):
    spamValue = 0
    hamValue = 0
    newDistinctValuefromTest = [] #to add new distinct words from test data
    for i in range(len(vocabList)):
        if(vocabList[i] in extractVocab):
            newDistinctValuefromTest.append(vocabList[i])
    for i in range(ClassCount):
        if(i == 0):
            spamValue = math.log10(float(prior[i]))
        else:
            hamValue = math.log10(float(prior[i]))
        for j in range(len(newDistinctValuefromTest)):
            if(i == 0):
                spamValue = spamValue + float(math.log10(spamVocabDict[str(newDistinctValuefromTest[j])]))
            else:
                hamValue = hamValue + float(math.log10(hamVocabDict[str(newDistinctValuefromTest[j])]))
                    
    finalValue = 0
    if(spamValue > hamValue):
        finalValue = 0
    else:
        finalValue = 1
    return finalValue

def trainNaiveBayes(ClassCount,trainingData, stopFile=None):
    if(stopFile != None):
        stopwords = ReadStopWordFile(stopFile)
        
    #preprocessing
    extractVocab = []
    docCount = 0 #storing total count of spam + ham
    docCountInClass = []
    mergeTextOfallDocsInClass = []
    stemmer = PorterStemmer()
    for i in range (len(trainingData)):
        mergeTextOfallDocsInClass.append([])
        td = os.listdir(trainingData[i])
        docCountInClass.append(len(td))#[spamCount,hamCount]
        docCount = docCount + len(td)#Spam + ham
        for j in range (docCountInClass[i]):
            txtFilePath = trainingData[i] + td[j] #fetching text file
            try:
                with open(txtFilePath, 'r', encoding='ascii', errors='ignore') as f:
                    for line in f:
                        line = line.translate(str.maketrans('', '', string.punctuation))
                        for word in line.split():
                            mergeTextOfallDocsInClass[i].append(stemmer.stem(word.lower()))#removed stemmer
                            extractVocab.append(stemmer.stem(word.lower()))
            except:
                print("Error while encoding the text file name:", txtFilePath)
            

    extractVocab = list(set(extractVocab))
    extractVocab.sort()
    
    if(stopFile != None):
        extractVocab = [x for x in extractVocab if x not in stopwords]
    
    extractVocabLen = len(extractVocab)
    
    spamVocabCountDict, hamVocabCountDict, spamVocabDict, hamVocabDict= {}, {},{}, {}
    denominator = 0
    prior = []
    
    for i in range (ClassCount):
        totalCountInOneClass = docCountInClass[i]#Spam or ham count
        prior.append(totalCountInOneClass/ float(docCount)) #Calculate prior probability

        data = mergeTextOfallDocsInClass[i]
        if(stopFile != None):
            data = [x for x in data if x not in stopwords]
        denominator = 0
        
        if(i == 0):
            denominator = len(data) + extractVocabLen
        else:
            denominator = len(data) + extractVocabLen
            
        for j in range (extractVocabLen):
            if(i == 0):
                txtCount = data.count(extractVocab[j]) #word freq
                spamVocabCountDict[extractVocab[j]] = txtCount
            else: 
                txtCount = data.count(extractVocab[j]) #word freq
                hamVocabCountDict[extractVocab[j]] = txtCount
        
        for k in range (extractVocabLen):
            if(i == 0):
                tmpProb = float(spamVocabCountDict[extractVocab[k]] + 1)/ float(denominator) # add one to numerator in order to avoid probability of word which dosent present in extractVocab list
                spamVocabDict.update({str(extractVocab[k]): tmpProb})
            else: 
                tmpProb = float(hamVocabCountDict[extractVocab[k]] + 1)/ float(denominator)
                hamVocabDict.update({str(extractVocab[k]): tmpProb})
    
    return extractVocab,prior,spamVocabDict,hamVocabDict
    
    
def MultinomialNaiveBayes(ClassCount,extractVocab,prior,spamVocabDict,hamVocabDict,Data):
    predict = 0
    totalCount = len(os.listdir(Data[0])) + len(os.listdir(Data[1]))

    for i in range (len(testData)):
        td = os.listdir(testData[i])
        for j in range (len(td)):
            txtFilePath = testData[i] + td[j] #fetching text file
            with open(txtFilePath, 'r', encoding='ascii', errors='ignore') as f:
                vocabList = []
                for line in f:
                    line = line.translate(str.maketrans('', '', string.punctuation))
                    for word in line.split():
                        vocabList.append(word)        
            finalValue = ApplyMultinomialNaiveBayes(ClassCount, extractVocab, prior, spamVocabDict, hamVocabDict, vocabList)
            if(finalValue == i):
                predict = predict + 1
    print("Accuracy training Data:", (predict*100)/float(totalCount))
    with open('result.txt','a') as f:
        f.write("Accuracy: " + str((predict * 100) / float(totalCount)))
#        print("Accuracy: " + str((predict * 100) / float(totalCount)),file=f)



#def Multinomial():
if __name__ == '__main__':
    if(len(sys.argv) != 4):
        print("Please enter:  python NaiveBayes.py train test Stopwords.txt")
        sys.exit()
        
    ClassCount = 2
    trainingData = [] #to store path of train data
    testData = [] #to store path of test data
    
#    
    trainingData.append(sys.argv[1] + "/spam/")
    trainingData.append(sys.argv[1] + "/ham/")
    testData.append(sys.argv[2] + "/spam/")
    testData.append(sys.argv[2] + "/ham/")
    
#    trainingData.append("train/spam/")
#    trainingData.append("train/ham/")
#    testData.append("test/spam/")
#    testData.append("test/ham/")
    
    with open('result.txt','w') as f:
        f.write("Navie Bayes Output file")
#        print("Navie Bayes Output file",file=f)
    
    print("Accuracy with stopwords")
    with open('result.txt','a') as f:
        f.write("Accuracy with stopwords")
#        print("Accuracy with stopwords",file=f)
    extractVocab,prior,spamVocabDict,hamVocabDict = trainNaiveBayes(ClassCount,trainingData)
    MultinomialNaiveBayes(ClassCount,extractVocab,prior,spamVocabDict,hamVocabDict,testData)
    
    print("\nAccuracy without stopwords")
    with open('result.txt','a') as f:
        f.write("\nAccuracy without stopwords")
#        print("\nAccuracy without stopwords",file=f)
    extractVocab,prior,spamVocabDict,hamVocabDict = trainNaiveBayes(ClassCount,trainingData,sys.argv[3])
    MultinomialNaiveBayes(ClassCount,extractVocab,prior,spamVocabDict,hamVocabDict,testData)

    print("\nAccuracy on train data")
    with open('result.txt','a') as f:
        f.write("\nAccuracy on train data")
#        print("\nAccuracy on train data",file=f)
    MultinomialNaiveBayes(ClassCount,extractVocab,prior,spamVocabDict,hamVocabDict,trainingData)
    
    print("Output is printed in result.txt file")
    f.close()

