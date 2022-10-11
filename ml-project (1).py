#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project - Iris Dataset
# 
# Use the "Run" button to execute the code.

# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


# Execute this to save new versions of the notebook
jovian.commit(project="ml-project")


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#You have to check difference between matplotlib and seaborn


# In[4]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression


# #### From seaborn load iris dataset and save it in iris dataframe >

# In[5]:


iris = sns.load_dataset('iris')


# In[6]:


iris


# #### Return the head of the dataframe >

# In[7]:


iris.head()


# #### Return the first 10 records of the dataframe >

# In[8]:


iris.head(10)


# #### Find the number of rows and columns in the dataset >

# In[9]:


iris.shape


# #### Find the number of unique species >

# In[10]:


iris['species'].nunique()


# #### Find the names of all unique species >

# In[11]:


iris['species'].unique()


# #### Find the number of unique petal_length >

# In[12]:


iris['petal_length'].nunique()


# #### Find the names of all unique species >

# In[13]:


iris['petal_length'].unique()


# #### Find the max and min value of petal_length >

# In[14]:


iris['petal_length'].max()


# In[15]:


iris['petal_length'].min()


# #### Find the mean of the petal_length >

# In[16]:


iris['petal_length'].mean()


# ## Data Visualization

# ### Scatter Plot 

# In[17]:


# Scatter plot
plt.scatter(iris['sepal_length'],iris['sepal_width'])

# For labeling axis with their names
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# For title
plt.title('Scatter plot on Iris dataset')


# #### Draw the scatter plot between petal length and petal width >

# In[18]:


plt.scatter(iris['petal_length'],iris['petal_width'])
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('PetalLength Vs. Petal width')


# #### Based on different species give different colors on the sepal length vs sepal width  >

# In[19]:


sns.set_style("whitegrid")
sns.FacetGrid(iris,hue="species")     .map(plt.scatter,"sepal_length","sepal_width")
plt.show()


# #### Change the set style in above plot. Also how will you figure out which color is for which species >

# In[20]:


sns.set_style("darkgrid")
sns.FacetGrid(iris,hue="species")     .map(plt.scatter,"sepal_length","sepal_width").add_legend() # Here add_legend is the function to determine color of the species
plt.show()


# #### Based on different species give different colors on the petal length vs petal width >

# In[21]:


sns.set_style("whitegrid")
sns.FacetGrid(iris,hue="species")     .map(plt.scatter,"petal_length","petal_width")
plt.show()


# #### Change the set style in above plot. Also how will you figure out which color is for which species >

# In[22]:


sns.set_style("whitegrid")
sns.FacetGrid(iris,hue="species")     .map(plt.scatter,"petal_length","petal_width").add_legend()
plt.show() 


# ### Pair Plot

# In[23]:


# Pair plot
sns.set_style("darkgrid")
sns.pairplot(iris,hue="species",height=2, diag_kind ="hist").add_legend()
plt.show() 


# ### Box Plot

# 

# #### Draw the boxplot for sepal length >

# In[24]:


sns.boxplot(y ="sepal_length", data = iris)


# #### Draw the boxplot for petal length >

# In[25]:


sns.boxplot(y ="petal_length", data = iris)


# #### Create boxplot for sepal length and different for different species >

# In[26]:


sns.boxplot(x="species",y ="sepal_length", data = iris)


# ### Kernal Distribution Estimate

# In[27]:


# Kernal Distribution Estimate
sns.FacetGrid(iris, hue = "species", height= 3)     .map(sns.kdeplot,"sepal_length")     .add_legend()
plt.show()


# ### Line Plot

# In[28]:


y = [1, 14, 3, 5, 4, 10]
plt.plot(y, label = "Line plot")
plt.xlabel('index')
plt.ylabel('value')
plt.legend()
plt.show()


# In[29]:


# Line plot
plt.plot(iris['sepal_length'], label ="Speal length")
plt.plot(iris['sepal_width'], label ="Speal width")
plt.plot(iris['petal_length'], label ="Petal length")
plt.plot(iris['petal_width'], label ="Petal width")
plt.legend()
plt.show()


# #### Figure out co relation between 2 values. Corelation lies between -1 <= cor <= 1
# 
# If coreation value is positive that means the relationship between x1 and x2 is increasing relationship......
# If coreation value is negative that means the relationship between x1 and x2 is decreasing relationship

# In[30]:


iris.corr()


# #### Plot about result as heatmap >

# In[31]:


sns.heatmap(iris.corr())


# #### Add annotations to heatmap >

# In[32]:


sns.heatmap(iris.corr(), annot = True)


# ### Histogram

# #### Draw histogram for sepal length

# In[33]:


plt.hist(iris['sepal_length'], bins = 5) # bins are which in the number in which data would be divided
plt.show()


# In[34]:


iris['sepal_length'].min()


# In[35]:


iris['sepal_length'].max()


# ### Seperating Data for training and testing

# #### X is training input set ( so here we are dropping species which is a outpur set and storing reamining to X) >

# In[36]:


X = iris.drop(['species'], axis = 1)


# In[37]:


X


# #### Y is training output set >

# In[38]:


Y = iris['species']


# In[39]:


Y


# In[40]:


# Imported this library above 
le = LabelEncoder()
Y = le.fit_transform(Y)
Y
# we got its encoding above as setosa:0, versicolor:1, verginica:2
# We encode thos because machine don't understand what setosa, verginica,etc means.


# #### Convert this X, Y to x_train, y_train, x_test, y_test >

# In[82]:


# Imported this library above 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
# test size 0.2 means 20% is the testing data and 80% is the training data chosen randomly


# #### Add random_state to the train_test_split >

# In[83]:


# to not choose data randomly that is don't shuffle
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)


# ### Logistic Regression

# In[84]:


# Imported this library above 
lr = LogisticRegression(solver = 'newton-cg')


# In[85]:


lr.fit(x_train, y_train)


# #### Get the prediction >

# In[86]:


y_pred1 = lr.predict(x_test)


# #### Get the confusion Matrix >

# In[87]:


# Imported this library above
# y_test = actual values , y_pred1= predicted data
confusion_matrix(y_test,y_pred1)


# #### Get the heatmap for confusion matrix >

# In[88]:


sns.heatmap(data=confusion_matrix(y_test,y_pred1), annot=True, cmap='Blues')
# here annot is to get values in the boxes
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')


# #### Get the accuracy score >

# In[89]:


# Imported this library above
accuracy_score(y_test, y_pred1)


# ### Decision Tree

# In[90]:


# imported the library above
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)


# #### Get the prediction >

# In[91]:


y_pred2 = dtree.predict(x_test)


# #### Get the confusion matrix >

# In[92]:


# Imported this library above
# y_test = actual values , y_pred2= predicted data
confusion_matrix(y_test,y_pred2)


# #### Get the heatmap for confusion matrix >

# In[93]:


sns.heatmap(data=confusion_matrix(y_test,y_pred2), annot=True, cmap='Blues')
# here annot is to get values in the boxes
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')


# #### Get the accuracy score >

# In[99]:


accuracy_score(y_test, y_pred2) # accuracy score is better than logistic regression


# In[98]:


iris.columns[:-1] # excluding species


# In[95]:


plt.figure(figsize = (20,20))
# Imported this library above
dec_tree = plot_tree(decision_tree=dtree, feature_names = iris.columns[:-1],
                    class_names = ["setosa", "vercicolor", "verginica"], filled = True)


# ### Random Forest

# In[110]:


# ensemble means collection of decision trees
from sklearn.ensemble import RandomForestClassifier


# In[137]:


clf = RandomForestClassifier(n_estimators = 2000)


# #### Use 100 decision trees in random forest >

# In[112]:


#clf = RandomForestClassifier(n_estimators = 100)


# In[138]:


clf.fit(x_train, y_train)


# #### Get the prediction >

# In[139]:


y_pred3 = clf.predict(x_test)


# #### Get the confusion matrix >

# In[140]:


# Imported this library above
# y_test = actual values , y_pred3= predicted data
confusion_matrix(y_test,y_pred3)


# #### Get the heatmap for confusion matrix >

# In[141]:


sns.heatmap(data=confusion_matrix(y_test,y_pred3), annot=True, cmap='Blues')
# here annot is to get values in the boxes
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')


# #### Get the accuracy score >

# In[142]:


accuracy_score(y_test, y_pred3)


# In[ ]:





# In[143]:


# Execute this to save new versions of the notebook
jovian.commit(project="ml-project")


# In[ ]:




