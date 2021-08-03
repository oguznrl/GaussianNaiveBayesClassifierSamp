import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from mlxtend.plotting import plot_decision_regions

X, y = make_blobs(n_samples=20, centers=[(0,0), (5,5), (-5, 5)], random_state=0)
plt.figure(figsize=(16, 8))
for i in range(3):
    plt.scatter(X[np.where(y==i), 0], X[np.where(y==i), 1], s=200, label=f'Class $c_{i+1}$')
plt.scatter([-2], [5], c='k', s=200)
plt.annotate('$x*$', (-1.9, 4.7))
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.style.use('fivethirtyeight')
plt.plot()
class GaussianNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def fit(self,X,y):
        X, y = check_X_y(X, y)
        self.priors_=np.bincount(y)/len(y)
        self.n_classes_=np.max(y)+1
        self.means_=np.array([X[np.where(y==i)].mean(axis=0) for i in range(self.n_classes_)])
        self.stds_=[X[np.where(y==i)].std(axis=0) for i in range(self.n_classes_)]
        return self
    
    def predict_prob(self,X):
        check_is_fitted(self)
        X = check_array(X)
        res=[]
        for i in range(len(X)):
            probas = []
            for j in range(self.n_classes_):
                probas.append((1/np.sqrt(2*np.pi*self.stds_[j]**2)*np.exp(-0.5*((X[i]-self.means_[j])/self.stds_[j])**2)).prod()*self.priors_[j])
            probas = np.array(probas)
            res.append(probas / probas.sum())
        return np.array(res)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        res = self.predict_prob(X)
        
        return res.argmax(axis=1)


my_gauss = GaussianNaiveBayesClassifier()
my_gauss.fit(X, y)
my_gauss.predict_prob([[-2, 5], [0,0], [6, -0.3]])
print(my_gauss.predict([[-2, 5], [0,0], [6, -0.3]]))

gnb = GaussianNB(var_smoothing=0)
gnb.fit(X, y)
gnb.predict_proba([[-2, 5], [0,0], [6, -0.3]])

print(gnb.predict([[-2, 5], [0,0], [6, -0.3]]))

plt.figure(figsize=(16, 8))
plot_decision_regions(X, y, clf=gnb, legend=0, colors='#1f77b4,#ff7f0e,#ffec6e')
for i in range(3):
    plt.scatter(X[np.where(y==i), 0], X[np.where(y==i), 1], s=200)
plt.scatter([-2, 0, 6], [5, 0, -0.3], c='k', s=200)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Decision Regions')
plt.plot()