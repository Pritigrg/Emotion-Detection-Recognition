
# coding: utf-8

# ## Emotion Detection and Recognition
# **<div style="text-align: right"> [Total score: 14]</div>**
# 
# We detected emotions from the text using scikit-learn in module-2 project. Now, we will use keras with word embeddings and LSTM layers to make the classification model.

# ### Ex1: Import Keras and other libraries
# Don't forget to import embedding layer and LSTM from Keras

# In[3]:


import numpy as np
import pandas as pd
# YOUR CODE HERE
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import string
import nltk

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


# ### Ex2: Load and Preprocess the Dataset
# **<div style="text-align: right"> [Score: 1]</div>**
# 1. Read the dataset ISEAR.csv which is in the current path.
# 2. Set the column names to Emotions and Sentence.
# 2. Visualize and clean the dataset.
# 3. Perform any preprocessing that may be useful
# 
# The data should possess only these emotions: 'joy', 'fear', 'anger', 'sadness', 'disgust', 'shame' and 'guilt'.

# In[4]:


# YOUR CODE HERE
column_names=['Emotions','Sentence']
df=pd.read_csv("./ISEAR.csv",names=column_names)
del df['Sentence']
df=df.reset_index()
df.rename(index=str,columns={"index":"Emotions","Emotions":"Sentence"},inplace= True)
df.head()





# In[5]:


df.describe()


# In[ ]:


#### INTENTIONALLY LEFT BLANK####


# In[23]:


X


# In[24]:


y


# ### Ex3: Clean the dataset
# **<div style="text-align: right"> [Score: 1]</div>**
# 1. Check if any values in the dataset contains null.
# 2. Drop all the null values if it exists.

# In[6]:


# YOUR CODE HERE
df.dropna(inplace=True)


# In[ ]:


#### INTENTIONALLY LEFT BLANK####


# In[7]:


print("The emotions to be classified are: "+ str(list(df.Emotions.unique().tolist())))


# Are you sure your data is clean ? <br>
# See if the spelling of guilt is incorrectly written as guit in some sentences.<br>
# **<div style="text-align: right"> [Score: 1]</div>**

# In[8]:


# YOUR CODE HERE
df['Emotions'].replace('guit','guilt',inplace=True)
print("We need to classify "+ str(df.Emotions.nunique()) +" emotions now.")
print("The emotions to be classified are: "+ str(list(df.Emotions.unique())))


# In[ ]:


#### INTENTIONALLY LEFT BLANK####


# In[9]:


from collections import Counter
Counter(df['Emotions'].values)


# ### Ex4: Further NLP based text Cleaning (Not Necessary to Implement all)
# You can try 
# - removing punctuations,
# - converting words to lower case 
# - using the stem of each word
# - other preprocessing techniques

# In[10]:


nltk.download("stopwords")


# In[11]:


def clean_text(text):
    # YOUR CODE HERE
    text=text.translate(string.punctuation)
    text=text.lower().split()
    
    from nltk.corpus import stopwords
    stops=set(stopwords.words('english'))
    text=[w for w in text if not w in stops]
    
    text=" ".join(text)
    
    
    
    
    return text

df['Sentence'] = df['Sentence'].map(lambda x: clean_text(x))


# In[12]:


X, y = df['Sentence'], df['Emotions']


# ### Ex5: Transform y to one-hot-encoding

# In[15]:


onehot_y = None
# YOUR CODE HERE
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
onehot_y = np_utils.to_categorical(encoded_y)


# In[16]:


vocab= set()
a= [vocab.add(el) for s in X.values for el in s.split(' ')]
print("Total Unique words:", len(a))


# In[17]:


l= [len(s) for s in X.values]
counts=Counter(l)
plt.bar(counts.keys(),counts.values())
plt.show()
print("Min:",min(l))
print("Median:",np.median(l))
print("Max:",max(l))


# ### Ex  6: Tokenize the sentences
# Tokenization of sentences is one of the essential parts in natural language processing. Tokenization simply divides a sentence into a list of words. 
# <br>We will use Keras tokenizer function to tokenize the strings and ‘texts_to_sequences’ to make sequences of words. You might also want to pad the sequences.

# In[18]:


vocabulary_size = 80896  # Select an Appropriate Vobabulary Size
padded_length = 150  # Select an Appropriate padded Length for text
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(X)
# YOUR CODE HERE


# In[20]:


def preprocessing(X):
    # YOUR CODE HERE
    sequences = tokenizer.texts_to_sequences(X)
    data=pad_sequences(sequences,maxlen = padded_length)
    
    
   
    return data
data = preprocessing(X)


# ### Ex7: Split the data into Train and Test
# Split the data into train and test with padded sequences as input and the encoded categorical y as the output; test size of 0.1 and random state of 101.

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(data,onehot_y,test_size=0.1,random_state=101,stratify=onehot_y)


# ### Ex8: Model and Compile your Neural Network Architecture
# **<div style="text-align: right"> [Score: 2]</div>**
# Try to Include 
# 1. Embedding layer
# 2. LSTM with dropout
# 3. Dense Output layere
# 4. Use any other layers as required

# In[37]:


model = None
learning_rate = 1e-2  # Choose your Learning Rate
numEpoch = 40  # choose Number of Epoch to train your model
embedding_size=200
lstm_size=64
seed=2019

import tensorflow as tf
import keras
from keras import optimizers,regularizers
from keras.layers import Dense,Flatten,LSTM,Dropout,Activation
from keras.layers.embeddings import Embedding
# YOUR CODE HERE
def create_model(embedding_size=100,lstm_size=32,layer1_size=32,dropoutrate=0.3):
    model=keras.models.Sequential()
    model.add(Embedding(vocabulary_size,embedding_size,input_length=padded_length,embeddings_regularizer=regularizers.l1(0.001)))
    model.add(LSTM(lstm_size,dropout=dropoutrate,recurrent_dropout=dropoutrate))
    model.add(Dense(layer1_size,activation ='relu'))
    model.add(Dropout(dropoutrate,seed=seed))
    model.add(Dense(7,activation ='softmax'))
    
    optim=optimizers.Adam(lr=learning_rate,beta_1=0.9,beta_2=0.999,epsilon=1e-08,
                         decay=learning_rate/numEpoch)
    model.compile(loss='categorical_crossentropy',optimizer=optim,metrics=['accuracy'])
    model.summary()
    return model
model=create_model(embedding_size=100,lstm_size=32,layer1_size=32,dropoutrate=0.3)        


# In[ ]:


#### INTENTIONALLY LEFT BLANK####


# ### Ex9: Set Callback functions and Train your Model
# Must include ModelCheckpoint

# In[40]:


checkpoint_path = 'best_model.h5'
from keras.callbacks import EarlyStopping,ModelCheckpoint
callbacks = [EarlyStopping(monitor='val_loss',patience=3),ModelCheckpoint(checkpoint_path,save_weights_only=True,save_best_only=True,verbose=True)]
# YOUR CODE HERE


# **Caution** Comment out the training section before submitting. Submit the code by loading Checkpoint model

# In[41]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def plot(history):
    """ Plots model loss from model train history"""
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper right')
    plt.show()
    
    # YOUR CODE HERE


# In[44]:


# Train your model
# #history = model.fit(X_train,y_train,batch_size=32,validation_split=0.2,
#                    epochs=numEpoch,callbacks=callbacks)
# YOUR CODE HERE

plot(history)


# ### Final Task
# **<div style="text-align: right"> [Score: 9]</div>**
# Comment out the previous training section and submit by loading your checkpoint below
# As the Dataset is small your test score and accuracy may be less but will probably perform better in hidden test set.
# Aim to achieve a score higher than 50 % or higher than your scikit learn project on the same dataset.

# In[45]:


train_model = model.load_weights(checkpoint_path)
score = model.evaluate(X_test, y_test) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])


# In[47]:


from sklearn import metrics
prediction=model.predict(X_test)

test_y=encoder.inverse_transform(np.argmax(y_test,axis=1,out=None))
pred_y=encoder.inverse_transform(np.argmax(prediction,axis=1,out=None))
print(metrics.classification_report(test_y,pred_y))


# In[49]:


cm=metrics.confusion_matrix(test_y,pred_y)
plt.figure(figsize=(7,7))
plt.imshow(cm,interpolation='nearest',cmap='Blues')
ticks=encoder.inverse_transform(range(7)).astype(str)
plt.xticks(range(7),ticks)
plt.yticks(range(7),ticks)


# In[ ]:


#### INTENTIONALLY LEFT BLANK####


# In[ ]:


#### INTENTIONALLY LEFT BLANK####


# In[ ]:


#### INTENTIONALLY LEFT BLANK####


# Congratulations, you have reached the end of the Assignment.
