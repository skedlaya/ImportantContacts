
#Program to predict important contacts

import csv
import json
import numpy as np

# Reading data
messages = []
# The columns are: accountId, messageId, threadId, from, to, cc, timestamp
with open('messages.csv', 'rb') as csvfile:
    msg_csvfile = csv.reader(csvfile, delimiter=',')
    for row in msg_csvfile:
        messages.append(row)
# Important contacts for account 1
with open('importantContacts_1.json') as json_data:
    contacts1 = json.load(json_data)
    json_data.close()
# Important contacts for account 2
with open('importantContacts_2.json') as json_data:
    contacts2 = json.load(json_data)
    json_data.close()

# Shuffle data to split into train and test data
np.random.shuffle(messages)

# Split into train data and test data
train_data = messages[:7*len(messages)/10]
test_data = messages[7*len(messages)/10:]

# Convert to contacts to set
contacts1 = set(contacts1)
contacts2 = set(contacts2)


# In[546]:

#Map from ID to a unique ID
fromList = set([d[3].strip() for d in train_data])
from_map = {}
i = 1
for m in fromList:
    from_map[(m)] = i
    i+=1


# In[547]:

# Map thread ID to number of messages in the thread
from collections import defaultdict
threadList = [int(d[2]) for d in train_data]
thread_map = defaultdict(int)
for t in threadList:
    thread_map[t] += 1


# In[548]:

# Input: A single message entry
# Returns: list of features extracted from the message
def feature(datum):
    # Bias
    feat = [1]
    
    # Account ID (0,1)
    feat.append(int(datum[0])-1)
    
    # Number of messages in the thread
    feat.append(thread_map[int(datum[2])])
    
    # One-hot encode email id category
    fList = [0]*len(fromList)
    try:
        fList[from_map[datum[3].strip()]] = 1
    except:
        pass
    feat = feat + fList
    
    # Number of to recipients
    len_to = len(datum[4].strip().split('|'))
    feat.append(len_to)
    
    # Number of cc recipients
    len_cc = len(datum[5].strip().split('|'))
    feat.append(len_cc)
    
    # cc recipient exists (0,1)
    if(len_cc>0):
        feat.append(1)
    else:
        feat.append(0)
        
    # Time - Same range as the other features
    feat.append(int(datum[6])/(10000000000))
    
    return feat


# In[549]:

# Input: A single message entry
# Returns: y value. important = 1, non important = 0
def map_y_data(datum):
    to_list = datum[4].strip().split("|")
    cc_list = datum[5].strip().split("|")
    to_list = [int(i) for i in to_list if i != '']
    cc_list = [int(i) for i in cc_list if i != '']
    account = int(datum[0])
    for n in to_list:
        if account == 1 and n in contacts1:
            return 1
        elif account == 2 and n in contacts2:
            return 1
    return 0


# In[550]:

# Extract features X from message
X_train = [feature(dat) for dat in train_data]
X_test = [feature(dat) for dat in test_data]


# In[551]:

# Extract prediction y from message
y_train = [map_y_data(dat) for dat in train_data]
y_test = [map_y_data(dat) for dat in test_data]


# In[602]:

# Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)
pred_test_RF = clf.predict(X_test)
pred_test_RF_proba= clf.predict_proba(X_test)
pred_test_RF_cont = []
for n in pred_test_RF_proba[:,1]:
    if n>0.3:
        pred_test_RF_cont.append(1)
    else:
        pred_test_RF_cont.append(0)


# In[595]:

# Logistic Regression Model
from sklearn import linear_model
clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
pred_test_LR= clf.predict(X_test)
pred_test_LR_proba= clf.predict_proba(X_test)
pred_test_LR_cont = []
for n in pred_test_LR_proba[:,1]:
    if n>0.3:
        pred_test_LR_cont.append(1)
    else:
        pred_test_LR_cont.append(0)


# In[606]:

# Evaluate results for Logistic Regression
pred_test_LR = np.array(pred_test_LR)
y_test = np.array(y_test)
matches =  sum([x == y for x,y in zip(pred_test_LR, y_test)])
fn = sum([y==1 and x==0 for x,y in zip(pred_test_LR, y_test)])
tn = sum([x==0 for x,y in zip(pred_test_LR, y_test)])
print "Logistic Regression, Binary Classification(Threshold = 0.5):"
print 'Classification accuracy (test) is: ' + str((matches*100.0)/len(y_test))
print 'False omission rate (test) is: ' + str(fn*1.0/tn)
print "\n"

pred_test_LR_cont = np.array(pred_test_LR_cont)
matches_cont =  sum([x == y for x,y in zip(pred_test_LR_cont, y_test)])
fn_cont = sum([y==1 and x==0 for x,y in zip(pred_test_LR_cont, y_test)])
tn_cont = sum([x==0 for x,y in zip(pred_test_LR_cont, y_test)])
print "Logistic Regression, Continuous Measure(Threshold = 0.3):"
print 'Classification accuracy (test) is: ' + str((matches_cont*100.0)/len(y_test))
print 'False omission rate (test) is: ' + str(fn_cont*1.0/tn_cont)
print "\n"


# In[607]:

# Evaluate results for Random Forest
pred_test_RF = np.array(pred_test_RF)
y_test = np.array(y_test)
matches =  sum([x == y for x,y in zip(pred_test_RF, y_test)])
fn = sum([y==1 and x==0 for x,y in zip(pred_test_RF, y_test)])
tn = sum([x==0 for x,y in zip(pred_test_RF, y_test)])
print "Random Forest Classification, Binary Classification(Threshold = 0.5):"
print 'Classification accuracy (test) is: ' + str((matches*100.0)/len(y_test))
print 'False omission rate (test) is: ' + str(fn*1.0/tn)
print "\n"

pred_test_RF_cont = np.array(pred_test_RF_cont)
matches_cont =  sum([x == y for x,y in zip(pred_test_RF_cont, y_test)])
fn_cont = sum([y==1 and x==0 for x,y in zip(pred_test_RF_cont, y_test)])
tn_cont = sum([x==0 for x,y in zip(pred_test_RF_cont, y_test)])
print "Random Forest Classification, Continuous Measure(Threshold = 0.3):"
print 'Classification accuracy (test) is: ' + str((matches_cont*100.0)/len(y_test))
print 'False omission rate (test) is: ' + str(fn_cont*1.0/tn_cont)
print "\n"

