#/bin/env python
#-*- coding=utf-8 -*-
#decision tree completement

import numpy as np
from collections import Counter, defaultdict
from functools import partial
import math, random, os

#Decision Tree class
class DecisionTree(object):
    def __init__(self, criterion='entropy', max_depth=3, random_state=None):
        np.random.seed(random_state)
        self.criterion = criterion
        self.max_depth=max_depth


    def fit(self, X, y, columns=None):
        self.treedepth=0
        #here, I used the number of list to represent the name of columns
        #maybe I can replace the real column name
        #self.column = np.arange(X.shape[0])
        if any(columns)==False:
            columns = range(X.shape[1])
        for col in columns:
            if '_' in col:
                print("Do not use the the name of %s with \"_\"" %(col))
                exit(0)
                
        self.columnname = columns
        #check the column names have "_" or not
        
        self.columndict = {attri:num
                           for attri, num in zip(columns,range(X.shape[1]))}
        #save all y labels
        self.ylabels = np.unique(y)
        #seperate each y to let Y have the same dimension with X in row
        Y = np.array([[yi] for yi in y])
        #combinate X and y for next compute (Y is in last column)
        combinedX = np.hstack((X,Y))

        self.tree = self.build_tree(X=combinedX)
        print ("Decision tree:", self.tree)
        return self.tree

    def predict(self, X):
        
        y=np.array([self.predict_single(xi,tree=self.tree) for xi in X])

        return y
    
    def predict_single(self, X, tree):
        #here, to predict the y with built tree
        #if a leaf node, return the value
        #input of X is for single vector
        if tree in self.ylabels:
            return tree

        #otherwise find the next subtree by the split
        attribute, subtree_dict = tree
        indexX = self.columndict[attribute]
        compare_value = X[indexX]
        #here I used '_' to get the criteria value by splitting
        criteria = np.unique([key.split('_')[2] for key in subtree_dict.keys()])[0] #should be only one value
        
        criteria = float(criteria)
        #default
        keystr='%s_gt_%f' %(attribute, criteria)
        if compare_value > criteria:
            keystr = '%s_gt_%08.3f' %(attribute, criteria)
        else:
            keystr = '%s_ls_%08.3f' %(attribute, criteria)
        subtree = subtree_dict[keystr]
        return self.predict_single(X, tree=subtree)
        
        
        
    def build_tree(self, X, split_candidates=None, depth=0):
        #first pass, the list of arange is used for split candidates
        y = X[:,-1] #retrieve the last column for use
        
        
        if split_candidates is None:
            split_candidates = self.columnname
        if len(np.unique(y))==1:
            return y[0] #if all the labels are the same

        num_inputs = X.shape[0]
        
        #if no split candidate can be used, return the most common element
        #if not split_candidates:
        #    label_counter = Counter(y)
        #    return label_counter.most_common(1)[0][0]

        #if we met the max depth limit
        if depth>self.max_depth:
            label_counter = Counter(y)
            return label_counter.most_common(1)[0][0]
        
        #start to split
        #select the best candidate to split the data
        #return [(criteria,impurity),(,).....]
        CriteImpur = [self.partition_criterion_by(X,attribute)
                      for attribute in split_candidates]
        #get the best_attribute
        Impudict = {atter:imp[1] for atter, imp in zip(split_candidates,CriteImpur)}
        best_attribute = min(Impudict.items(), key= lambda t:t[1])[0]
        best_criteria = min(CriteImpur, key= lambda t:t[1])[0]
        partitions = self.partition_by(X,attribute=best_attribute,criteria=best_criteria)
        
        #binary class
        #new_candidates = [cname for cname in split_candidates
        #                  if cname!=best_attribute]
        #multiplt class, so the previous best_attribute is keeped
        new_candidates = [cname for cname in split_candidates]

        #build the subtrees recursively
        subtrees = { key:self.build_tree(np.array(subset),new_candidates, depth+1)
                     for key, subset in partitions.items()}
        
        return (best_attribute, subtrees)

    def partition_by(self, X, attribute, criteria):
        #return a defaultdict(list), each item represent a situation that
        #greater (gt) or less and equal (ls) than criteria
        groups= defaultdict(list)
        for xi in X:
            value = xi[self.columndict[attribute]]
            if value > criteria:
                key= "%s_gt_%08.3f" %(attribute,criteria)
                groups[key].append(xi)
            else:
                key= "%s_ls_%08.3f" %(attribute,criteria)
                groups[key].append(xi)
        return groups

    def partition_criterion_by(self, X, attribute):
        """find the minimun impurities of each attributes
        for the assigned criterion, and return two dictionaries.
        first is impurities, second is criteria
        """
        indexX = self.columndict[attribute]
        #{ci:partition_impurity(criteria) for criteria, ci in enumerate(X[:,indexX])}
        impur_list=[]
        for criteria in X[:, indexX]:
            partitions = self.partition_by(X, attribute, criteria)
            impur_list.append((criteria,self.partition_impurity(partitions.values())))

        #return (criteria, min impurity)
        best_criteria = min( impur_list, key=lambda t:t[1]) 
        return best_criteria

    def partition_impurity(self, subsets):
        total_count = sum(len(subset) for subset in subsets)
        Ids = [] 
        for subset in subsets:
            yindex = len(subset[0])-1
            labels = [arr[yindex] for arr in subset]
            probabilities = self.class_probabilities(labels)
            
            if (self.criterion=='entropy'):
                Ids.append(self.cri_entropy(probabilities))
            elif(self.criterion=='gini'):
                Ids.append(self.cri_gini(probabilities))
            elif(self.criterion=='classerror'):
                Ids.append(self.cri_classerror(probabilities))
            else: #default is entropy
                Ids.append(self.cri_entropy(probabilities))
                
        #Ids is a 1-dim list, len is the same with subsets
        return sum(Id*len(subset)/total_count for Id, subset in zip(Ids, subsets))
    
    def class_probabilities(self,labels):
        total_count = len(labels)
        return [count/total_count
                for count in Counter(labels).values()]
        
    def cri_entropy(self,class_probabilities):
        "return entropy impurity"
        return sum(-p*np.log2(p)
                   for p in class_probabilities
                   if p)
    def cri_gini(self,class_probabilities):
        "return Gini impurity"
        return 1-sum(p*p
                     for p in class_probabilities
                     if p)
    def cri_classerror(self,class_probabilities):
        "return classification error impurity"
        return 1-max(p
                     for p in class_probabilities
                     if p)

    
        
