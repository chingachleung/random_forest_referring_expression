import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

np.random.seed(42)
facts = pd.read_csv("gum_refexp_data.tab", sep="\t", quoting=3)

facts["mentioned_bin"] = np.where(facts["mentioned"]=="yes",1,0)
facts["label_bin"] = np.where(facts["label"]=="pronoun",1,0)

facts["ordinal_entity"] = np.where((facts["entity"]=="person")|(facts["entity"]=="organization")|
                                   (facts["entity"]=="object")|(facts["entity"]=="event")|
                                   (facts["entity"]=="animal"),1,2)

#interaction between entity types and distance 
facts["distance_entity_product"] = facts["ordinal_entity"] * facts["distance"] 
#brief observation shows that the below functions are more likely for lexical label
facts["ordinal_next_func"] =np.where((facts["next_func"].str.contains("list"))|(facts["next_func"].str.contains("compound"))|
                                      (facts["next_func"].str.contains("conj"))|(facts["next_func"].str.contains("dep"))|
                                      (facts["next_func"].str.contains("root"))|(facts["next_func"].str.contains("acl"))|
                                      (facts["next_func"].str.contains("advmod")),2,1)
facts["ordinal_func"] =np.where((facts["func"].str.contains("list"))|(facts["func"].str.contains("compound"))|
                                      (facts["func"].str.contains("conj"))|(facts["func"].str.contains("dep"))|
                                      (facts["func"].str.contains("root"))|(facts["func"].str.contains("acl"))|
                                      (facts["func"].str.contains("advmod")),2,1)

#while func itself does not have much impact on the accuracy, the compound effect of the two funcs seems to have an impact
facts["funcs_product"] = facts["ordinal_func"] * facts["ordinal_next_func"]


#try out the numerical features on logistic regression
features = ["mentioned_bin","position","distance","ordinal_entity","funcs_product","distance_entity_product","ordinal_next_func","ordinal_func"]

# Get train and dev
from sklearn.preprocessing import OrdinalEncoder
train = facts.loc[facts["partition"] == "train"]
dev = facts.loc[facts["partition"] == "dev"]
X = train[features]
y = train["label_bin"]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#baseline model using logistic regression 
reg = LogisticRegression()
reg.fit(X,y)
preds = reg.predict(dev[features])

import statsmodels.discrete.discrete_model as sm
model = sm.Logit(y, X)
model.fit().summary()

#improved random forrest model
cat_features = ["entity","mentioned","genre","cpos"]
num_features =  ["position","funcs_product","distance_entity_product","ordinal_next_func","ordinal_entity"]
encoder = OrdinalEncoder()
train = facts.loc[facts["partition"] == "train"]
dev = facts.loc[facts["partition"] == "dev"]

train_ord = encoder.fit_transform(train[cat_features].reset_index(drop=True))
train_rf = pd.concat([pd.DataFrame(train_ord,columns=cat_features),train[num_features].reset_index(drop=True)],axis=1)
train_rf["len_head"] = train["head"].str.len().reset_index(drop=True)
dev_ord = encoder.transform(dev[cat_features].reset_index(drop=True))
dev_rf = pd.concat([pd.DataFrame(dev_ord,columns=cat_features),dev[num_features].reset_index(drop=True)],axis=1)
dev_rf["len_head"] = dev["head"].str.len().reset_index(drop=True)


rf = RandomForestClassifier(random_state=42, n_estimators = 100, n_jobs=3)
rf.fit(train_rf,train["label_bin"])

preds = rf.predict(dev_rf)
print(accuracy_score(dev["label_bin"],preds))
