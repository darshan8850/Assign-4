import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from autograd import grad, jacobian
from tqdm import tqdm

style.use('seaborn')

class BinaryLogisticReg:
    def __init__(self, n_feat, use_bias=False):
        self.use_bias = use_bias
        if self.use_bias:
            self.theta = np.random.normal(size=(n_feat+1, 1))
        else:
            self.theta = np.random.normal(size=(n_feat, 1))

    def train(self, X, Y, lamb=0.0, lr=1e-4, EPOCH=1, return_loss=False):

        def binary_cross_entropy_loss(y, y_hat, eps=1e-6):
            return np.mean( 
                    np.multiply(-1, 
                        np.add( np.multiply(y, np.log(y_hat + eps)), 
                            np.multiply( np.subtract(1, y), 
                                np.log(np.subtract(1, y_hat) + eps)
                            )
                        )
                    )
                )
        
        def sigmoid(X):
            return np.divide(1.0, np.add( 1.0, np.exp( np.multiply(-1.0, X) ) ) )

        def fit(theta):
            H = np.dot(X_hat, theta)
            Z = sigmoid( H )
            loss = binary_cross_entropy_loss(Y, Z) + lamb * np.sqrt(np.sum(np.square(theta)))
            return loss

        if self.use_bias:
            X_hat = np.insert(X, 0, np.ones(shape=(X.shape[0],)), axis=1)
        else:
            X_hat = X
        
        grad_fn = grad(fit)

        tq = tqdm(range(1, EPOCH+1))

        loss_list = []
        itr_list  = []

        for epoch in tq:
            loss        = fit(self.theta)
            self.theta -= lr * grad_fn(self.theta)
            # print('Train Epoch: {}\t Loss: {}'.format(epoch, loss))
            tq.set_description('Train Loss [{}]'.format(loss))
            itr_list.append(epoch)
            loss_list.append(loss)
        
        if return_loss:
            return itr_list, loss_list
    
    def test(self, X):
        
        def sigmoid(X):
            return np.divide(1.0, np.add( 1.0, np.exp( np.multiply(-1.0, X) ) ) )

        if self.use_bias:
            X_hat = np.insert(X, 0, np.ones(shape=(X.shape[0],)), axis=1)
        else:
            X_hat = X

        H = np.dot(X_hat, self.theta)
        Z = sigmoid( H )
        Z = np.where(Z < 0.5, 0, 1)
        return Z
    
    def draw_decision_sruface(self, X, Y, split='Train'):
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        h      = 50
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
        z  = np.reshape(self.test(np.c_[xx.ravel(), yy.ravel()]), xx.shape)
        plt.contourf(xx, yy, z, cmap='viridis', alpha=0.5)
        plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=1.0, edgecolors='k')
        plt.title('Logistic Regression {}'.format(split))
        plt.tight_layout()
        plt.show()



class MultiClassLogisticReg:
    def __init__(self, n_feat, n_class, use_bias=False):
        self.use_bias = use_bias
        self.n_class  = n_class
        if self.use_bias:
            self.theta = 1.0/np.random.normal(size=(n_feat+1, n_class))
        else:
            self.theta = 1.0/np.random.normal(size=(n_feat, n_class))

    def train(self, X, Y, lamb=0.0, lr=1e-4, EPOCH=1, return_loss=False):

        # def binary_cross_entropy_loss(y, y_hat, eps=1e-6):
        #     return np.mean( 
        #             np.multiply(-1, 
        #                 np.add( np.multiply(y, np.log(y_hat + eps)), 
        #                     np.multiply( np.subtract(1, y), 
        #                         np.log(np.subtract(1, y_hat) + eps)
        #                     )
        #                 )
        #             )
        #         )
        
        # def sigmoid(X):
        #     return np.divide(1.0, np.add( 1.0, np.exp( np.multiply(-1.0, X) ) ) )
        def softmax(X):
            X_temp = X - X.max()
            return np.divide(
                                np.exp( X_temp ),
                                np.sum(np.exp( X_temp ), axis=-1, keepdims=True)
                            )
        
        def fit(theta):
            # print(X_hat.shape, theta.shape)
            H = np.matmul(X_hat, theta)
            # Z = sigmoid( H )
            Z = softmax(H)

            Y_temp = np.array(list(zip(range(0, Z.shape[0]), Y.tolist())))

            loss   = -1 * np.mean(
                            np.log(Z[Y_temp[:, 0], Y_temp[:, 1]] + 1e-6)
                                , axis=-1, keepdims=True) + lamb * np.sqrt(np.sum(np.square(theta), axis=-1))
            # loss   = np.mean(loss)
            # cls_loss = 0.0
            # for c in cls:
            #     Y_temp    = np.where(Y == c, 1, 0)
            #     cls_loss += binary_cross_entropy_loss(Y_temp, Z)
            # loss = cls_loss + lamb * np.sqrt(np.sum(np.square(theta)))

            return loss

        if self.use_bias:
            X_hat = np.insert(X, 0, np.ones(shape=(X.shape[0],)), axis=1)
        else:
            X_hat = X
        
        # grad_fn = grad(fit)
        grad_fn = jacobian(fit)

        tq = tqdm(range(1, EPOCH+1))

        loss_list = []
        itr_list  = []

        for epoch in tq:
            loss        = np.mean(fit(self.theta))
            # print(loss)
            for tape in grad_fn(self.theta):
                # print(tape)
                self.theta -= lr * tape
                # break
            # print('Train Epoch: {}\t Loss: {}'.format(epoch, loss))
            tq.set_description('Train Loss [{}]'.format(loss))
            itr_list.append(epoch)
            loss_list.append(loss)
        
        if return_loss:
            return itr_list, loss_list
    
    def test(self, X):
        
        def sigmoid(X):
            return np.divide(1.0, np.add( 1.0, np.exp( np.multiply(-1.0, X) ) ) )

        if self.use_bias:
            X_hat = np.insert(X, 0, np.ones(shape=(X.shape[0],)), axis=1)
        else:
            X_hat = X

        H = np.dot(X_hat, self.theta)
        Z = np.divide(
                    np.exp( H ),
                    np.sum(np.exp( H ), axis=-1, keepdims=True)
                )
        Y_cap = np.argmax(Z, axis=-1)
        return Y_cap
    
    def draw_decision_sruface(self, X, Y, split='Train'):
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        M            = len(np.unique(Y))
        plt.figure(figsize=(9, 9))
        # for c, cls in enumerate(np.unique(Y)):
        h      = 50
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
        # ax = plt.subplot2grid((1, 3), (0, c), rowspan=1, colspan=1)
        z  = np.reshape(self.test(np.c_[xx.ravel(), yy.ravel()]), xx.shape)
        plt.contourf(xx, yy, z, cmap='viridis', alpha=0.5)
        # ax.scatter(X[:, 0], X[:, 1], c=Y, alpha=1.0, edgecolors='k')
        # ax.set_title('Multiclass Logisitic Regression [class: {}]'.format(cls))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.8, edgecolors='k')
        plt.title('Mutliclass Classification Iris Dataset')
        # plt.legend()
        plt.tight_layout()
        plt.show()