

### importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

import warnings
warnings.filterwarnings('ignore')


### loading the dataset
df = pd.read_csv('name_gender.csv')
df.head()

### checking the shape of dataset
#print(df.shape)


### checking the missing values
#print(df.isnull().sum())  # there are no missing values



### checking the 
#v = df['M'].value_counts()
#print(v)


# text preprocessing
#df.Aaban = df.Aaban.apply(lambda x: str(x).lower())
df.M[df.M == 'M'] = 1  # adding m = 1
df.M[df.M == 'F'] = 0  # adding F = 0

###############
name  = list(df['Aaban'])
labels = list(df['M'])



## Converting word to vector
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(analyzer='char')

names=cv.fit_transform(name).toarray()

# print(len(names[1])) # 26 

### splitting data into train and test split
from sklearn.model_selection import train_test_split
feature_train,feature_test,label_train,label_test=train_test_split(names,labels,test_size=0.2,random_state=42)


# Modeling

### Applying MultinomailNB
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(feature_train,label_train)

# predict the test set
label_pred=model.predict(feature_test)

# Evaluating the peroformance of model
import sklearn.metrics as m
print(m.accuracy_score(label_test,label_pred))

###########################################################################################
# Applying LSTM model


from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(char_level=True)  #To convert text into numbers we have a class in keras called Tokenizer.
tokenizer.fit_on_texts(name)  # # fit_on_text method accepts only list
sequence_of_int = tokenizer.texts_to_sequences(name)   # # it converts the generated tokens in the sequence.


## adding pad sequences
from keras.preprocessing.sequence import pad_sequences
padsequences=pad_sequences(sequence_of_int,maxlen=15,padding='post')
#print(len(padsequences[2]))

#### converting into cateogory
from keras.utils.np_utils import to_categorical
labels=to_categorical(labels)

#### Loading all essential  layer 
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding,Dropout
#print(padsequences.shape)


### Again splitting data into train and test set
from sklearn.model_selection import train_test_split
feature_train,feature_test,label_train,label_test=train_test_split(padsequences,labels,test_size=0.1,random_state=42)

# creating LSTM
model=Sequential()
model.add(Embedding(27,64,input_length=15))
model.add(LSTM(2048,return_sequences=True))
model.add(LSTM(256,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# fitting the model
model.fit(feature_train,label_train,epochs=10,validation_data=(feature_test,label_test),batch_size=1000)

# saving the model
model.save('model.h5')


### predict the test set
label_pred=model.predict(feature_test)

label_pred=np.argmax(label_pred,axis=1)
#print(label_pred)

### 
l_test=np.argmax(label_test,axis=1)


# Evaluating the model performance
print(m.accuracy_score(l_test,label_pred))  # acc. 87 percent but we can increase by using more epoch



from keras.models import load_model
import pickle


pickle.dump(tokenizer,open('tokenizer.pkl','wb+'),protocol=pickle.HIGHEST_PROTOCOL)

tokenizer=pickle.load(open('tokenizer.pkl','rb+'))

#print(tokenizer.index_word)


## function will return sequence of text in numeric form
def input(n):
  q=tokenizer.texts_to_sequences(n)
  q=[i[0] for i in q]
  q1=[]
  q1.append(q)
  s=pad_sequences(q1,maxlen=15,padding='post')
  return s


#print(tokenizer.texts_to_sequences('Aabid'))
#print(input('Aabid'))


## lets  create a small function for testing the work
def pred(n):
  w=model.predict(input(n))
  w=np.argmax(w,axis=1)
  return w[0]

pred('Aadan')  # correct prediction Male(1)

pred('Aahana')  # correct prediction female(0)


################################
# Future developemnt : We can increase performance of the model by incresing the epoch size & fine tuning




