from abc import abstractmethod
from dataclasses import dataclass
from random import randrange
import subprocess
from nptyping import NDArray
import dcor
from sklearn.neighbors import NearestNeighbors
from data_efficient_dependency_estimation.dependency_tests_thesis.XtendedCorrel import hoeffding
from tensorflow.math import digamma
import numpy as np
import pandas as pd

@dataclass
class DependencyMeasure():
    
    @abstractmethod
    def apply(self,samples):
        return NotImplementedError

    def variance(self):
        return 0

@dataclass
class dHSIC(DependencyMeasure):
    
    def apply(self, samples):
        samples = np.asarray([ele for ele in samples if len(ele) != 0])
        t = self.dHSIC(samples,samples)
        return t

    """
    Hilbert Schmidt Information Criterion with a Gaussian kernel, based on the
    following references
    [1]: https://link.springer.com/chapter/10.1007/11564089_7
    [2]: https://www.researchgate.net/publication/301818817_Kernel-based_Tests_for_Joint_Independence
    """
    def centering(self,M):
        """
        Calculate the centering matrix
        """
        n = M.shape[0]
        unit = np.ones([n, n])
        identity = np.eye(n)
        H = identity - unit / n

        return np.matmul(M, H)


    def gaussian_grammat(self, x, sigma=None):
        """
        Calculate the Gram matrix of x using a Gaussian kernel.
        If the bandwidth sigma is None, it is estimated using the median heuristic:
        ||x_i - x_j||**2 = 2 sigma**2
        """
        try:
            x.shape[1]
        except IndexError:
            x = x.reshape(x.shape[0], 1)

        xxT = np.matmul(x, x.T)
        xnorm = np.diag(xxT) - xxT + (np.diag(xxT) - xxT).T
        if sigma is None:
            #fix size of xnorm
            mdist = np.median(xnorm)
            sigma = np.sqrt(mdist * 0.5)

        # --- If bandwidth is 0, add machine epsilon to it
        if sigma == 0:
            eps = 7. / 3 - 4. / 3 - 1
            sigma += eps

        KX = - 0.5 * xnorm / sigma / sigma
        np.exp(KX, KX)
        return KX


    def dHSIC_calc(self, K_list):
        """
        Calculate the HSIC estimator in the general case d > 2
        """
        if not isinstance(K_list, list):
            K_list = list(K_list)

        n_k = len(K_list)

        length = K_list[0].shape[0]
        term1 = 1.0
        term2 = 1.0
        term3 = 2.0 / length

        for j in range(0, n_k):
            K_j = K_list[j]
            term1 = np.multiply(term1, K_j)
            term2 = 1.0 / length / length * term2 * np.sum(K_j)
            term3 = 1.0 / length * term3 * K_j.sum(axis=0)

        term1 = np.sum(term1)
        term3 = np.sum(term3)
        dHSIC = (1.0 / length) ** 2 * term1 + term2 - term3
        return dHSIC


    def HSIC(self, x, y):
        """
        Calculate the HSIC estimator for d=2, as in [1] eq (9)
        """
        n = x.shape[0]
        return np.trace(np.matmul(self.centering(self.gaussian_grammat(x)), self.centering(self.gaussian_grammat(y)))) / n / n


    def dHSIC(self, *argv):
        assert len(argv) > 1, "dHSIC requires at least two arguments"

        if len(argv) == 2:
            x, y = argv
            return self.HSIC(x, y)

        K_list = [self.gaussian_grammat(_arg) for _arg in argv]

        return self.dHSIC_calc(K_list)
@dataclass
class Hoeffdings(DependencyMeasure):

    def apply(self, samples: NDArray):
        samples = np.squeeze(samples)
        x, y = samples[:,0], samples[:,1] 
        r = hoeffding(x,y)
        return r

@dataclass
class dCor(DependencyMeasure):

    def apply(self, samples: NDArray):
        samples = np.squeeze(samples)
        x, y = samples[0], samples[1:] 
        r = dcor.distance_correlation(samples,samples)
        return r
@dataclass
class CMI(DependencyMeasure):

    def apply(self, samples):
        x = np.asarray([item.tolist() for sublist in samples for item in sublist])

        dataFile = 'run_data_store/cmiData.csv'
        df = pd.DataFrame(x)
        df.to_csv(dataFile, sep=",", header='true', index=False)

        #output = subprocess.check_output('java -jar target/scala-2.12/MCDE-experiments-1.0.jar -t EstimateDependency -f src/test/resources/data/Independent-5-0.0.csv -a CMI -m 1 -p 1')
        output = subprocess.check_output('java -jar target/scala-2.12/MCDE-experiments-1.0.jar -t EstimateDependency -f run_data_store/cmiData.csv -a CMI')
        score = float(output.splitlines()[5])
        
        return score

@dataclass
class HiCS(DependencyMeasure):

    def apply(self, samples: NDArray):
        x = np.asarray([item.tolist() for sublist in samples for item in sublist])
        dataFile = 'run_data_store/hicsData.csv'
        df = pd.DataFrame(x)
        df.to_csv(dataFile, sep=",", header='true', index=False)

        output = subprocess.check_output('java -jar target/scala-2.12/MCDE-experiments-1.0.jar -t EstimateDependency -f run_data_store/hicsData.csv -a HiCS')
        score = float(output.splitlines()[5])
        return score

@dataclass
class MCDE(DependencyMeasure):

    def apply(self, samples: NDArray):
        x = np.asarray([item.tolist() for sublist in samples for item in sublist])
        dataFile = 'run_data_store/mcdeData.csv'
        df = pd.DataFrame(x)
        df.to_csv(dataFile, sep=",", header='true', index=False)

        #output = subprocess.check_output('java -jar target/scala-2.12/MCDE-experiments-1.0.jar -t EstimateDependency -f src/test/resources/data/Independent-5-0.0.csv -a MWP -m 1 -p 1')
        output = subprocess.check_output('java -jar target/scala-2.12/MCDE-experiments-1.0.jar -t EstimateDependency -f run_data_store/mcdeData.csv -a MWP -m 1 -p 1')
        score = float(output.splitlines()[5])
        return score

@dataclass
class IMIE(DependencyMeasure):

    OrderR = []
    OrderX = []
    OrderY = []
    xP = []
    yP = []
    k: int = 5
    offset: float = 0
    mean: float = 0 
    var: float = 0 
    iterations: int = 0

    def apply(self,samples):
        self.xP = samples[:,:1]
        self.yP = samples[:,1:2]
        self.offset = digamma(self.xP.size()) + digamma(self.k) - (1.0 / float(self.k))
        self.OrderR = list(range(len(samples)))
        self.OrderX = list(range(len(samples)))
        self.OrderY = list(range(len(samples)))
        for i in range(10):
            score, v = self.incrementEstimate()
        return score

    def deltaX(self,index):
        q = self.exp_modules.queried_data_pool.queries[index].reshape(-1, 1)       
        #get k nearest for p
        knn = NearestNeighbors(n_neighbors=self.k)
        knn.fit(self.exp_modules.queried_data_pool.queries, self.exp_modules.queried_data_pool.results)
        kneighbor_indexes = knn.kneighbors(q, n_neighbors=self.k, return_distance=False)
        kneighbors = self.exp_modules.queried_data_pool.results[kneighbor_indexes]
        xP = self.xP[index]

        return max([abs(xP[0]-x[0]) for x in kneighbors[0]])

    def deltaY(self,index):
        q = self.exp_modules.queried_data_pool.queries[index].reshape(-1, 1)       
        #get k nearest for p
        knn = NearestNeighbors(n_neighbors=self.k)
        knn.fit(self.exp_modules.queried_data_pool.queries, self.exp_modules.queried_data_pool.results)
        kneighbor_indexes = knn.kneighbors(q, n_neighbors=self.k, return_distance=False)
        kneighbors = self.exp_modules.queried_data_pool.results[kneighbor_indexes]
        yP = self.yP[index]

        return max([abs(yP[0]-y[0]) for y in kneighbors[0]])


    def MC(self,index):
        xP = self.xP[index]
        yP = self.yP[index]
        a = len([elem for elem in self.xP if abs(elem - xP) <= self.deltaX(index)])
        b = len([elem for elem in self.yP if abs(elem - yP) <= self.deltaY(index)])
        return (a,b)
    
    def incrementEstimate(self):
        randomIdx = randrange(len(self.xP))
        self.OrderR[self.iterations], self.OrderR[randomIdx] = self.OrderR[randomIdx], self.OrderR[self.iterations]
        index = self.OrderR[self.iterations]
        MCs = self.MC(index)
        v = digamma(float(MCs[0])).numpy() + digamma(float(MCs[1])).numpy()
        self.iterations+=1
        delta = v - self.mean
        self.mean += (delta / (self.iterations))
        delta2 = v - self.mean
        self.var += (delta * delta2)

        score = self.offset - self.mean
        return score

    def variance(self):
        return self.var / self.iterations
