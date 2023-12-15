# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 06:44:44 2023

@author: Adrian Curic

research
"""

#Read SGML using BeautifulSoup #used also for html parsing 
def read_reuters_sgm(location = param_location):
    '''Read SGM files containing Reuters annotated article fomat
    Return a list of dictionaries corresponding to the parsed articles 
    '''
    res_list = []
    for file in os.listdir(location):
        if file.endswith(".sgm"): 
            filename = os.path.join(location, file)
            f = open(filename, 'r', encoding='utf-8', errors='ignore')
            dataFile = f.read()
            soup = BeautifulSoup(dataFile, 'html.parser')
            for item in soup.findAll('reuters'): 
                res = {}
                res['bodies'] = item.find("body")
                res['topics'] = item.find("topics")
                res['lewissplit'] = item.find('lewissplit')
            res_list.append(res)
            break
    return res_list

def read_reuters_sgm(location = param_location):
    '''Read SGM files containing Reuters annotated article fomat
    Return a list corresponding to the parsed articles 
    '''
    bodies = []
    topics = []
    lewissplit  = []
    file_list = [e for e in os.listdir(location) if e.endswith(".sgm")]    
    for file in tqdm(file_list):
        filename = os.path.join(location, file)
        #print(filename)
        f = open(filename, 'r', encoding='utf-8', errors='ignore')
        dataFile = f.read()
        soup = BeautifulSoup(dataFile, 'html.parser')
        for item in soup.findAll('reuters'): 
            bodies.append(item.find("body"))
            topics.append(item.find("topics"))
            lewissplit.append(item['lewissplit'])
            
        #break
    return list(zip(bodies,topics,lewissplit))

def read_reuters_sgm(location = param_location):
    '''Read SGM files containing Reuters annotated article fomat
    Return a list corresponding to the parsed articles 
    '''
    bodies = []
    topics = []
    lewissplit  = []
    file_list = [e for e in os.listdir(location) if e.endswith(".sgm")]    
    for file in tqdm(file_list):
        filename = os.path.join(location, file)
        #print(filename)
        f = open(filename, 'r', encoding='utf-8', errors='ignore')
        dataFile = f.read()
        soup = BeautifulSoup(dataFile, 'html.parser')
        for item in soup.findAll('reuters'): 
            bodies.append(item.find("body"))
            topics.append(item.find("topics"))
            lewissplit.append(item['lewissplit'])
            #Title, PLACES, people, orgs, EXCHANGES, COMPANIES, AUTHOR
                        
        #break
    return list(zip(bodies,topics,lewissplit))

# %%

if False:
    #Traditional preprocessing used in the article:
    res_list = read_reuters_sgm()
    reuters_df = pd.DataFrame(res_list, columns=["articles","topic","lewissplit"])
    reuters_df.dropna(inplace=True)
    # converting articles and topic columns to their text format
    reuters_df["articles"]=reuters_df["articles"].apply(lambda x: x.text)
    reuters_df["topic"]=reuters_df["topic"].apply(lambda x: x.text)
    
    print(reuters_df["topic"].value_counts(normalize=True))

# %%

# %%

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
x = tfidf.transform(xa["articles"].tolist())
mask_train = xa["lewissplit"]=="TRAIN"
r_train = xa.loc[mask_train, encode_col].to_numpy()
r = xa.loc[:, encode_col].to_numpy()
X_train_plus = np.concatenate([X_train.toarray(), r_train], axis = 1)
X_plus = np.concatenate([x.toarray(), r], axis = 1)
model.fit(X_train_plus, y_train)
pred = model.predict(X_plus) #predict for everything
xa['topic_pred_plus'] = pred
y_test = xa.loc[~mask_train, 'topic_simple']
y_pred = xa.loc[~mask_train, 'topic_pred_plus']


# get the multi-label confusion matrix
mcm = multilabel_confusion_matrix(y_test, y_pred,
            labels= topic_list)

cm = confusion_matrix(y_test, y_pred,
            labels= topic_list)

print(classification_report(y_test, y_pred))


# %%
    
# Display all individual confusion matrix
# Multi-label MCM is [[TN, FP], [FN, TP]]
# Ordinary confusion matrix is [[TP,FP],[FN,TN]]

def mcm_to_cm(mcm):
    '''
    # Multi-label MCM is [[TN, FP], [FN, TP]] - TN and TP are switched
    # Ordinary confusion matrix is [[TP,FP],[FN,TN]]
    '''
    cm = np.array([[mcm[1,1], mcm[0,1]],[mcm[1,0],mcm[0,0]]])
    return cm
          
        
    cm_display  = ConfusionMatrixDisplay(confusion_matrix = ecm, display_labels = [False, True])
    res = cm_display.plot()
    res.title(f'{label} - confusion matrix')
    cm_display.show()
    
    print(label, e)
    
# %%




# Encode 

    
reuters_df_train = reuters_df[reuters_df["lewissplit"]=="TRAIN"]
reuters_df_test  = reuters_df[reuters_df["lewissplit"]=="TEST"]


# %% Read file directly to save time:

if True:
    #read from csv    
    reuters_df = pd.read_csv('reuters_processed.csv')
    df = reuters_df
    df.topics = df.topics.apply(eval)
    
    
# %% Follow the rest of the article 

# Data cleaning 
reuters_df["articles_cln"]=reuters_df["articles"].apply(lambda x:clean_articles(x))

reuters_df_earn=reuters_df[reuters_df["topic"]=="earn"]
reuters_df_others=reuters_df[reuters_df["topic"]!="earn"]

# Data Transformation
# segregating the data into train and test dataframes
#initializing the vectorizer.
tfidf = TfidfVectorizer(min_df=5)
# vectorizing train and test datasets
X_train = tfidf.fit_transform(reuters_df_train["articles"].tolist())
X_test  = tfidf.transform(reuters_df_test["articles"].tolist())
# coverting the labels to binary- 1 for earn and 0 for others
y_train = reuters_df_train["topic"].apply(lambda x: 1 if x=="earn" else 0).tolist()
y_test  = reuters_df_test["topic"].apply(lambda x: 1 if x=="earn" else 0).tolist()
print(f"The vectorizer has {X_train.shape[0]} rows and {X_train.shape[1]} features")

# %%

def make_confusion_matrix(y_actual,y_predict,type_data):
    cm = metrics.confusion_matrix(y_actual, y_predict)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['others', 'earn']
    plt.title(f'Confusion Matrix - {type_data} Data')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)

    for i in range(2):
        for j in range(2):
            plt.text(j,i,str(cm[i][j]))
    plt.show()
    print(f"\n Classification Matrix for {type_data} data:\n",metrics.classification_report(y_actual, y_predict))
    
##  Function to calculate different metric scores of the model - Accuracy, Recall and Precision
def get_metrics_score(model,flag=True):
    '''
    model : classifier to predict values of X

    '''
    # defining an empty list to store train and test results
    score_list=[] 
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    train_acc = model.score(X_train,y_train)
    test_acc = model.score(X_test,y_test)
    
    train_recall = metrics.recall_score(y_train,pred_train)
    test_recall = metrics.recall_score(y_test,pred_test)
    
    train_precision = metrics.precision_score(y_train,pred_train)
    test_precision = metrics.precision_score(y_test,pred_test)  
    
    train_f1 = metrics.f1_score(y_train,pred_train)
    test_f1 = metrics.f1_score(y_test,pred_test)  
    
    score_list.extend((train_acc,test_acc,train_recall,test_recall,train_precision,test_precision,train_f1,test_f1))
        
    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True: 
        print("Accuracy on training set : ", model.score(X_train,y_train))
        print("Accuracy on test set : ",     model.score(X_test,y_test))
        print("\nRecall on training set : ", metrics.recall_score(y_train,pred_train))
        print("Recall on test set : ",       metrics.recall_score(y_test,pred_test))
        print("\nPrecision on training set : ", metrics.precision_score(y_train,pred_train))
        print("Precision on test set : ",     metrics.precision_score(y_test,pred_test))
        print("\nF1 score on training set : ", metrics.f1_score(y_train,pred_train))
        print("F1 score on test set : ",     metrics.f1_score(y_test,pred_test))
        
    
    return score_list # returning the list with train and test scores

# %% M1 Model 

# Try logistic regression
logr_model=LogisticRegression(random_state=42)
logr_model.fit(X_train, y_train)
LogisticRegression(random_state=42)
logr_score= get_metrics_score(logr_model)

y_pred_train=logr_model.predict(X_train)
y_pred_test=logr_model.predict(X_test)
## training data
make_confusion_matrix(y_train,y_pred_train,"train")
## training data
make_confusion_matrix(y_test,y_pred_test,"test")

# %% M2 model

# Try SVC 
svc_model = LinearSVC(random_state=42)
svc_model.fit(X_train, y_train)
svc_score=get_metrics_score(svc_model)
y_pred_train=svc_model.predict(X_train)
y_pred_test=svc_model.predict(X_test)
## training data
make_confusion_matrix(y_train,y_pred_train,"train")
## training data
make_confusion_matrix(y_test,y_pred_test,"test")

# Skip the results score 
#Maybe use your seaborn to print stuff
# TODO - set up a gradio ?? to allow classification

