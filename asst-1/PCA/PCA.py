
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# In[24]:


img=Image.open('test.jpeg')
img=np.array(img)
img=img.astype('float64')
a,b,c=img.shape



# In[26]:


flat=img.reshape(-1,3)
flat=flat.T
flat[0].shape


# In[27]:


def center(X):
    for i in range(len(X)):
        X[i]=X[i]-np.mean(X[i])
    return X


# In[28]:


flat1=center(flat)
print(np.mean(flat1[0]))
cov=np.matmul(flat1,flat1.T)
cov=cov/len(flat1[0])


# In[29]:


values,matrix=np.linalg.eig(cov)
y=np.matmul(matrix.T,flat)


# In[32]:

output=y.T
print(output.shape)
# shape=(a,b,2)
output=output.reshape(a,b,c)
output=output.astype('int')
plt.imshow(output)

plt.show()
# In[33]:
