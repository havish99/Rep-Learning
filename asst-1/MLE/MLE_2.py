import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
class MLE:
	def __init__(self,N_points,N_bins,mean,std):
		self.N_points=N_points  # number of training samples
		self.N_bins=N_bins      # number of columns in histogram
	def hist_plot(self,generated):
		fig, axis = plt.subplots(1, 3, sharey=True, tight_layout=True)
		axis[0].hist(self.Ground_truth, bins=self.N_bins,alpha=0.6,label="Ground Truth")
		axis[0].legend()
		axis[1].hist(generated, bins=self.N_bins,alpha=0.6,label="generated")
		axis[1].legend()
		axis[2].hist(generated,bins=self.N_bins,alpha=0.6,label="generated")
		axis[2].hist(self.Ground_truth,bins=self.N_bins,alpha=0.6,label="Ground Truth")
		axis[2].legend()
		plt.show()
	def Gaussian_MLE(self):
		self.Ground_truth=np.random.normal(mean,std,N_points)
		generated_mean=np.mean(self.Ground_truth)
		generated_std=np.std(self.Ground_truth)
		generated=np.random.normal(generated_mean,generated_std,self.N_points)
		return generated
	def Laplacian_MLE(self):
		self.Ground_truth=np.random.normal(mean,std,N_points)
		generated_mu=np.median(self.Ground_truth)
		generated_b=np.mean(np.abs(self.Ground_truth-generated_mu))
		generated=np.random.laplace(generated_mu,generated_b,self.N_points)
		return generated
	def exponential_MLE(self):
		self.Ground_truth=np.random.normal(mean,std,N_points)
		generated_beta=np.mean(self.Ground_truth)
		generated=np.random.exponential(generated_beta,self.N_points)
		return generated
	def poisson_MLE(self):
		self.Ground_truth=np.random.normal(mean,std,N_points)
		generated_lambda=np.mean(self.Ground_truth)
		generated=np.random.poisson(generated_lambda,self.N_points)
		return generated
	def binomial_MLE(self):
		num_trials=int(raw_input("Enter number of trials: "))
		self.Ground_truth=np.random.randint(0,num_trials,self.N_points)
		generated_prob=np.mean(self.Ground_truth)/num_trials*1.0
		generated=np.random.binomial(num_trials,generated_prob,self.N_points)
		return generated

N_points=10000
N_bins=100
mean=100
std=10
mle=MLE(N_points,N_bins,mean,std)
while(1):
	choice=int(raw_input("Enter choice of distribution: "))
	if(choice==1):
		generated=mle.Gaussian_MLE()
		mle.hist_plot(generated)
	if(choice==2):
		generated=mle.Laplacian_MLE()
		mle.hist_plot(generated)
	if(choice==3):
		generated=mle.exponential_MLE()
		mle.hist_plot(generated)
	if(choice==4):
		generated=mle.poisson_MLE()
		mle.hist_plot(generated)
	if(choice==5):
		generated=mle.binomial_MLE()
		mle.hist_plot(generated)
