import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def form_clusters(centroids,img):
    diction={}
    for i in range(0,len(centroids)):
        diction[i]=[]
    a,b,c=img.shape
    img=img.reshape(a*b,c)
    dist=[]
    for centroid in centroids:
        dist.append(np.linalg.norm((img-centroid),axis=1))
    dist=np.array(dist)
    dist1=dist.T
    for i in range(0,len(dist1)):
        diction[np.argmin(dist1[i])].append(img[i])
    return diction


def training(centroids,img,thresh):
    err=thresh+1
    while(err>thresh):
        new_centroids=[]
        diction=form_clusters(centroids,img)
        for i in range(0,len(centroids)):
            if(len(diction[i])!=0):
                new_centroids.append(np.mean(diction[i],axis=0))
            else:
                new_centroids.append(centroids[i])
        # new_centroids=centroids(diction)
        new_centroids=np.array(new_centroids)
        err=np.linalg.norm(new_centroids-centroids)
        print(err)
        centroids=new_centroids
    return centroids

def initialize(img,num_clusters):
    K=[]
    for i in range(num_clusters):
        K.append(img[np.random.randint(len(img)),np.random.randint(len(img[0]))])
    K=np.array(K)
    return K

def KMeans_test(img,centroids,cluster_number):
    diction={}
    shape=img.shape
    img1=img.reshape(-1,3)
    d=np.full(img1.shape,0)
    dist=[]
    for i in range(0,len(centroids)):
        diction[i]=[]
    for elem in centroids:
        dist.append(np.linalg.norm((img1-elem),axis=1))
    dist=np.array(dist)
    dist=dist.T
    mini=np.argmin(dist,axis=1)
    for i in range(len(mini)):
        if(mini[i]==cluster_number):
            d[i]=img1[i]
    d=d.reshape(shape)
    d=d.astype("int")
    print("Elements of the cluster: ")
    print(d)
    plt.imshow(d)
    plt.show()


np.random.seed(42)
img=Image.open("strips.jpg")
img=np.array(img)
img=img.astype("float64")
num_clusters=int(raw_input("Enter number of clusters: "))
threshold=float(raw_input("Enter threshold: "))
K=initialize(img,num_clusters)
centroids=training(K,img,threshold)
print("Centroids after convergence: ")
print(centroids)
while(1):
    number=int(raw_input("Enter cluster number: "))
    KMeans_test(img,centroids,number)
