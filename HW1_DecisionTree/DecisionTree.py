import pandas as pd
from collections import Counter
from pprint import pprint
import math
import sys

def ReadCSVFile(path):
    dataset = pd.read_csv(path)
    dataset.head()
    return dataset
    
def EntropyOfDataSet(trainingset):
    count = Counter(x for x in trainingset)
    TotalInstances = len(trainingset)*1.0
    probs = [x / TotalInstances for x in count.values()]
    entropy = Entropy(probs)
    return entropy
    
def Entropy(probs):
    probSum = 0
    for prob in probs:
        probSum +=  -prob*math.log(prob, 2)
        
    return probSum

def findCount(seq, return_counts=False, id=None):
   
    found = set()
    if id is None:
        for x in seq:
            found.add(x)
           
    else:
        for x in seq:
            x = id(x)
            if x not in found:
                found.add(x)
    found = list(found)           
    counts = [seq.count(0),seq.count(1)]
    if return_counts:
        return found,counts
    else:
        return found
     
def TotalCount(data):
    total = 0
    for i in data:
        total = total + i
    return total

def calculate_variance(target_values):
    values = list(target_values)
    elm,counts = findCount(values,True)
    variance_impurity = 0
    totalInstances = TotalCount(counts)
    for i in elm:
        variance_impurity += (-counts[i]/totalInstances*(counts[i]/totalInstances))
    return variance_impurity

def Variance_gain(dataset, targetAttribute, colName):
    ds_split = dataset.groupby(colName)
    nos = len(dataset.index)*1.0
    df_agg_var = ds_split.agg({targetAttribute : [calculate_variance, lambda x: len(x)/nos] })[targetAttribute]
    df_agg_var.columns = ['Variance', 'Observations']
    variance_impurity = sum( df_agg_var['Variance'] * df_agg_var['Observations'] )
    total_variance_impurity = calculate_variance(dataset[targetAttribute])
    variance_impurity_gain = total_variance_impurity - variance_impurity
    return variance_impurity_gain

def InformationGain(dataset, targetAttribute, colName):
    dS_Split = dataset.groupby(colName)
    nos = len(dataset.index)*1.0
    df_agg_ent = dS_Split.agg({targetAttribute : [EntropyOfDataSet, lambda x: len(x)/nos] })[targetAttribute]
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    old_entropy = EntropyOfDataSet(dataset[targetAttribute])
    return old_entropy - new_entropy

def findRootNode(dataset, targetAttribute, columnsNames, default_Class=None):
    
    count = Counter(x for x in dataset[targetAttribute])
    
    if len(count) == 1:
        return next(iter(count))
    elif dataset.empty or (not columnsNames):
        return default_Class
    else:
        default_Class = max(count.keys())
        gain = [InformationGain(dataset, targetAttribute, colName) for colName in columnsNames]
    
        index_of_max = gain.index(max(gain))
        
        best_Node = columnsNames[index_of_max]
        
        tree = {best_Node : {}}
        rem_ColumnNames = [i for i in columnsNames if i != best_Node]
    
    
    for colValue, subDataset in dataset.groupby(best_Node):
        subTree = findRootNode(subDataset, targetAttribute, rem_ColumnNames, default_Class)
        tree[best_Node][colValue] = subTree
        
    return tree

def buildTreeUsingVariance(dataset, targetAttribute, columnsNames, default_Class=None):
    
    count = Counter(x for x in dataset[targetAttribute])
    
    if len(count) == 1:
        return next(iter(count))
    elif dataset.empty or (not columnsNames):
        return default_Class
    else:
        default_Class = max(count.keys())
        gain = [Variance_gain(dataset, targetAttribute, colName) for colName in columnsNames]
    
        index_of_max = gain.index(max(gain))
        
        best_Node = columnsNames[index_of_max]
        
        tree = {best_Node : {}}
        rem_ColumnNames = [i for i in columnsNames if i != best_Node]
    
    
    for colValue, subDataset in dataset.groupby(best_Node):
        subTree = buildTreeUsingVariance(subDataset, targetAttribute, rem_ColumnNames, default_Class)
        tree[best_Node][colValue] = subTree
        
    return tree


def accuracy_of_the_tree(instance, tree, default=None):
    attribute = list(tree.keys())[0]
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict): 
            return accuracy_of_the_tree(instance, result)
        else:
            return result 
    else:
        return default 
    

if(len(sys.argv) != 6):
    sys.exit("Please provide input in format: .\ProgramName <training-set> <validation-set> <test-set> <to-print> to-print:{yes,no} <prune> prune:{yes:no}")
else:
    trainingData = ReadCSVFile(sys.argv[1])
    validationData = ReadCSVFile(sys.argv[2])
    testData = ReadCSVFile(sys.argv[3])
    doPrint = sys.argv[4]
    doPrune = sys.argv[5]
    attributeNames = list(trainingData.columns)
    attributeNames.remove('Class')
    print("Kindly wait to see the end result:\n")
    tree = findRootNode(trainingData, 'Class', attributeNames)
    varTree = buildTreeUsingVariance(trainingData, 'Class', attributeNames)
    
    if(doPrint == 'yes'):
        print("Final Tree with Information Gain:\n")
        pprint(tree)
        print("Final tree with Variance Gain:\n")
        pprint(varTree)
    #findLeafNode(tree)
    
    if(doPrune == 'yes'):
        print("Not able to complete pruning.\n")
        
        
    #Accuracy on Training data:H1 is information Gain
    print("Dataset1 or Dataset2\n")
    print("\nAccuracy with Information Gain:\n")
    #tree = findRootNode(trainingData, 'Class', attributeNames,"IG")
    trainingData['predictTrainData'] = trainingData.apply(accuracy_of_the_tree, axis=1, args=(tree,'1')) 
    print( 'H1 NP Training ' +  (str( sum(trainingData['Class']==trainingData['predictTrainData'] ) / (0.01*len(trainingData.index)) )) + "%")
    
    #Accuracy on Validation data::H1 is information Gain
    attributeNames = list(validationData.columns)
    attributeNames.remove('Class')
    tree = findRootNode(validationData, 'Class', attributeNames)
    validationData['predictValData'] = validationData.apply(accuracy_of_the_tree, axis=1, args=(tree,'1')) 
    print( 'H1 NP Validation ' +  (str( sum(validationData['Class']==validationData['predictValData'] ) / (0.01*len(validationData.index)) )) + "%")
    
    #Accuracy on Test data:H1 is information Gain
    attributeNames = list(testData.columns)
    attributeNames.remove('Class')
    tree = findRootNode(testData, 'Class', attributeNames)
    testData['predictTestData'] = testData.apply(accuracy_of_the_tree, axis=1, args=(tree,'1')) 
    print( 'H1 NP Test ' +  (str( sum(testData['Class']==testData['predictTestData'] ) / (0.01*len(testData.index)) )) + "%")
    
    #Accuracy on Training data:H1 is information Gain
    print("\nAccuracy with Variance Gain:\n")
    #varTree = findRootNode(trainingData, 'Class', attributeNames, "Var")
    trainingData['predictTrainData'] = trainingData.apply(accuracy_of_the_tree, axis=1, args=(varTree,'1')) 
    print( 'H2 NP Training ' +  (str( sum(trainingData['Class']==trainingData['predictTrainData'] ) / (0.01*len(trainingData.index)) )) + "%")
    
    #Accuracy on Validation data::H1 is information Gain
    attributeNames = list(validationData.columns)
    attributeNames.remove('Class')
    varTree = buildTreeUsingVariance(validationData, 'Class', attributeNames)
    validationData['predictValData'] = validationData.apply(accuracy_of_the_tree, axis=1, args=(varTree,'1')) 
    print( 'H2 NP Validation ' +  (str( sum(validationData['Class']==validationData['predictValData'] ) / (0.01*len(validationData.index)) )) + "%")
    
    #Accuracy on Test data:H1 is information Gain
    attributeNames = list(testData.columns)
    attributeNames.remove('Class')
    varTree = buildTreeUsingVariance(testData, 'Class', attributeNames)
    testData['predictTestData'] = testData.apply(accuracy_of_the_tree, axis=1, args=(varTree,'1')) 
    print( 'H2 NP Test ' +  (str( sum(testData['Class']==testData['predictTestData'] ) / (0.01*len(testData.index)) )) + "%")
    
    

    