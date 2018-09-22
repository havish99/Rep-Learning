import numpy as np


np.random.seed(42)

def datagen(d,n):
    mean=np.linspace(0,1,d)
    covar=np.identity(d)
    X=np.random.multivariate_normal(mean,covar,n)
    print(X.shape)
    return X

class GMM():
    def __init__ (self,X,k,d,n):
        self.X=X
        self.k=k
        self.d=d
        self.n=n
        self.w=np.ones(k)
        self.w=self.w/k
        self.mean=np.random.normal(0,1,(k,d))
        self.covar=[]
        for i in range(k):
            num=np.identity(d)
            self.covar.append(num)
        self.covar=np.array(self.covar)

    def normal(self,i,k):
        x=self.X[i]
        mean=self.mean[k]
        covar=self.covar[k]
        det=np.linalg.det(covar)
        inv=np.linalg.inv(covar)
        nr=np.exp(-0.5*(np.matmul((np.matmul(x-mean,inv)),(x-mean).T)))
        dr=np.sqrt(det*((2*np.pi)**(len(x))))
        return nr/dr

    def posterior(self,i):
        tot=[]
        for j in range(self.k):
            temp=self.w[j]*self.normal(i,j)
            tot.append(temp)
        tot=np.array(tot)
        post=tot/np.sum(tot)
        return post,tot

    def likelihood(self):
        normals=[]
        for i in range(self.n):
            _,tmp=self.posterior(i)
            normals.append(tmp)
        normals=np.array(normals)
        normals=np.log(normals)
        L=np.sum(normals)
        return L

    def fit(self):
        iter=0
        while(iter<100):
            post=[]
            for i in range(self.n):
                tmp,_=self.posterior(i)
                post.append(tmp)
            post=np.array(post)
            post=post.reshape(self.n,self.k)
            Nk=np.sum(post,axis=0)
            Nk=Nk.reshape(len(Nk),1)
            numer=np.dot(post.T,self.X)
            self.mean=numer/Nk
            for i in range(self.k):
                temp=(post.T)[i]
                temp=temp.reshape(len(temp),1)
                a=temp*(self.X-self.mean[i])
                self.covar[i]=np.matmul(a.T,self.X-self.mean[i])
            self.w=Nk/self.n
            new_L=L=self.likelihood()
            print(self.mean)
            iter=iter+1

X=datagen(2,100)
GMM=GMM(X,4,2,100)
GMM.fit()
