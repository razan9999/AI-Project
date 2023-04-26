#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import pandas as pdd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler


# In[118]:


data = pd.read_csv(r"C:\Users\razan\OneDrive\Desktop\Razan Desktop\AI Project\Cancer_Data.csv")
cancer1= pd.read_csv(r"C:\Users\razan\OneDrive\Desktop\Razan Desktop\AI Project\Cancer_Data.csv")
print(data)


# In[119]:


cancer1.shape


# In[120]:


cancer1.columns


# In[121]:


cancer1.info()


# In[122]:


#the first 10 rows of the dataset.
cancer1.head(10)


# In[157]:


cancer1['diagnosis'].replace(['B','M'],[0, 1],inplace=True)


# In[128]:


cancer1.head()


# In[131]:


cancer1['diagnosis'].value_counts()
#distrbution plot of texture mean 
sns.displot(
  data=cancer1,
  x="texture_mean",
  aspect=1)


# In[133]:


#distribution plot of texture se 
sns.displot(
  data=cancer1,
  x="texture_se",
  aspect=1
)


# In[136]:


#bar plot 
pic = sns.countplot(data=cancer1, x='diagnosis')
plt.title('Total Benign x Malignant cells')
pic.bar_label(pic.containers[0])
plt.show()


# In[147]:


# Scatter graphs

cancer2=cancer1[['diagnosis', 'radius_mean', 'radius_se', 'radius_worst']]

sns.pairplot(cancer2,hue='diagnosis')


# In[145]:


cancer1.isnull()



# In[146]:


cancer1.describe()


# In[68]:


data['area_mean'].max


# In[156]:


# Separate the aimed variable from the Variables 
aim1 = cancer1 ['diagnosis']
var1 = cancer1.drop('diagnosis', axis=1)
# Normalize the Variables using the z-score method
var1 = (var1 - var1.mean()) / var1.std()
# Combine the normalized features with the target variable
normalized_cancer1 = pd.concat ([var1, aim1], axis=1)
# Print the normalized data
print (normalized_cancer1.head())


# In[148]:


data['radius_mean'] = pd.cut(x = data['radius_mean'], bins=[0,10,20,30], right= False, 
                            labels=[1, 2, 3])


# In[149]:


data.head()


# In[100]:


diagnosis = ['diagnosis']
radius_mean = ['radius_mean']
texture_mean = ['texture_mean']
perimeter_mean = ['perimeter_mean']
area_mean = ['area_mean']
texture_worst = ['texture_worst']
perimeter_worst = ['perimeter_worst']
area_worst = ['area_worst']
smoothness_worst = ['smoothness_worst']
compactness_worst = ['compactness_worst']
concavity_worst = ['concavity_worst']
concavePoints_worst = ['concave points_worst']
symmetry_worst=['symmetry_worst']
fractal_dimension_worst= ['fractal_dimension_worst']



# In[151]:


data.boxplot(radius_mean)



# In[154]:


data.boxplot(texture_mean)


# In[153]:


data.boxplot(perimeter_mean)


# In[53]:


data.boxplot(area_mean)


# In[152]:


data.boxplot(texture_worst)


# In[59]:


data.boxplot(perimeter_worst)


# In[60]:


data.boxplot(area_worst)
area_worst = ['area_worst']
smoothness_worst = ['smoothness_worst']
compactness_worst = ['compactness_worst']
concavity_worst = ['concavity_worst']
concavePoints_worst = ['concave points_worst']
symmetry_worst=['symmetry_worst']
fractal_dimension_worst= ['fractal_dimension_worst']


# In[61]:


data.boxplot(smoothness_worst)


# In[62]:


data.boxplot(compactness_worst)


# In[63]:


data.boxplot(concavity_worst)


# In[64]:


data.boxplot(concavePoints_worst)


# In[65]:


data.boxplot(symmetry_worst)


# In[66]:


data.boxplot(fractal_dimension_worst)


# In[ ]:





# In[ ]:




