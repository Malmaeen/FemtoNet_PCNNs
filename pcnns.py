import sys,os
# -- local
import tools as tl
# -- matplotlib
from matplotlib import pyplot as plt
# --numpy
import numpy as np
# -- sklearn
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# -- pandas
import pandas as pd

#--tensorflow
import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers        as kl
import tensorflow.keras.models        as km
import tensorflow.keras.optimizers    as ko
import tensorflow.keras.callbacks     as kc
import tensorflow.keras.regularizers  as kr
#from tensorflow.keras import Model

import matplotlib
matplotlib.rc('text',usetex=True)

class PCNNS():
    
    def __init__(self, outdir='.'):
        
        tl.checkdir(outdir)
        self.outdir=outdir
        self.input_shape = (5) 
        self.output_shape = 1
        self.nodes = 1024
        self.batch_size = 512
        self.epochs = 1500
        self.Drate= 0.2
        self.l2_reg  = 1e-4
        self.BATCH_SIZE= 512
        self.PHI=  4
        self.model_name = 'pcnn'
        self.opt = ko.Adam(learning_rate=1e-4)

        
    def normalized_data(self):  
        param, param_rep,param_test, sigma, sigma_rep, sigma_test= tl.data_loader()
        self.param_sc       =  MinMaxScaler() 
        self.param_norm     =  self.param_sc.fit_transform(param)
        self.param_rep_norm =  self.param_sc.transform(param_rep)  
        
        self.sigma_sc       =  QuantileTransformer(output_distribution='normal')
        self.sigma_norm     =  self.sigma_sc.fit_transform(sigma.reshape(-1,1))
        self.sigma_rep_norm =  self.sigma_sc.transform(sigma_rep.reshape(-1,1))
        
    def split_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.param_rep_norm, 
                                                                                self.sigma_rep_norm, test_size=0.1, random_state=42)
     
            
    def create_pcnns(self):
        act  = kl.LeakyReLU(alpha=0.1)
        inputs = kl.Input(shape = self.input_shape, name='pcnn_input')

        x1= kl.Dense(self.nodes , activation = act, input_dim= self.input_shape)(inputs)
        x2= kl.Dropout(self.Drate)(x1)
        x3= kl.Dense(self.nodes , activation = act, kernel_regularizer=kr.l2(self.l2_reg))(x2)
        x4= kl.Dropout(self.Drate)(x3)
        x5= kl.Dense(self.nodes , activation = act, kernel_regularizer=kr.l2(self.l2_reg))(x4)
        x6= kl.Dropout(self.Drate)(x5)

        yy= kl.Dense(self.output_shape)(x6)
        model= km.Model(inputs=inputs, outputs= yy)
        model. compile(loss='mse',optimizer= self.opt)
        return model
        
    def plot_loss(self,Mloss, Sloss, Closs):
            Mloss= np.array(Mloss)
            Sloss= np.array(Sloss)
            Closs= np.array(Closs)


            plt.figure(figsize=(10,6))
            plt.plot(np.arange(Mloss.shape[0]), Mloss, label= "MSE")
            plt.plot(np.arange(Sloss.shape[0]), Sloss,label= "Symmetric loss")
            plt.plot(np.arange(Closs.shape[0]), Closs, label="Edges loss ")
            plt.ylabel("Loss",size=30)
            plt.xlabel("Epoch",size=30)
            plt.xticks(size=20)
            plt.yticks(size=20)
            plt.legend(fontsize=20)
            plt.show()
            plt.tight_layout()

    # -- Train the model
    def train(self):
        model = self.create_pcnns()
        loss=[]
        loss_mse_= []
        loss_sym_=[]
        loss_closs_=[]
        print("Training started: ")
        print("==============================")
        for epoch in range(self.epochs):
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
            loss = 0.0
            loss_sym = 0.0
            loss_close = 0.0

            for i in range(self.X_train.shape[0]//self.BATCH_SIZE):
                loss = loss + model.train_on_batch(self.X_train[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE], self.y_train[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE])

            for j in range(8):
                X = np.copy(self.X_train)       

                X[:, self.PHI] = np.random.uniform(0, 1, self.X_train.shape[0])

                XX = np.copy(X)
                XX[:, self.PHI] = 1.0 - XX[:, self.PHI]
                YY = model.predict(XX)
                for i in range(self.X_train.shape[0]//self.BATCH_SIZE):
                    loss_sym = loss_sym + model.train_on_batch(X[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE], YY[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE])

            for k in range(5):
                X_head = np.copy(self.X_train)
                X_head[:, self.PHI] = 0
                y_head = model.predict(X_head)
                X_tail = np.copy(self.X_train)
                X_tail[:, self.PHI] = 1
                y_tail = model.predict(X_tail)

                for i in range(self.X_train.shape[0]//self.BATCH_SIZE):
                    loss_close = loss_close + model.train_on_batch(X_head[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE], y_tail[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE])
                    loss_close = loss_close + model.train_on_batch(X_tail[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE], y_head[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE])

            loss = loss/(self.X_train.shape[0]//self.BATCH_SIZE)
            loss_sym = loss_sym/(self.X_train.shape[0]//self.BATCH_SIZE)
            loss_close = loss_close/(self.X_train.shape[0]//self.BATCH_SIZE)
            loss_mse_.append(loss)
            loss_sym_.append(loss_sym)
            loss_closs_.append(loss_close)

            print("epoch = ", epoch, " loss = ", loss, " loss_sym = ", loss_sym, " loss_close = ", loss_close)
            model.save_weights('%s/%s.hdf5'%(self.outdir,self.model_name))


        self.plot_loss(loss_mse_,loss_sym_,loss_closs_)
        return model    
    
    
    def plot_result_1(self):
        model= self.create_pcnns()
        model.load_weights('%s/%s.hdf5'%(self.outdir,self.model_name))
        pred=  model.predict(self.X_test)
        pred  = self.sigma_sc.inverse_transform(pred)
        true=   self.sigma_sc.inverse_transform(self.y_test)
        
        plt.plot(true,true,'.',label=r'$\rm True $')
        plt.plot(true,pred,'.',alpha=0.4,label= r'$\rm PCCNNs $')
        plt.legend(fontsize=15)
        plt.semilogx()
        plt.semilogy()
        plt.show()
        
    

        
    def plot_result_2(self):
        model= self.create_pcnns()
        model.load_weights('%s/%s.hdf5'%(self.outdir,self.model_name))
        extrap_Example= tl.Ex_test()
        extrap_Example_norm= self.param_sc.transform(extrap_Example)
        pred_array = []
        for i in range(100):
            res = model(extrap_Example_norm, training=True)
            pred_array.append(self.sigma_sc.inverse_transform(res))
        pred = np.asarray(pred_array)
        pred=pred[:,:,0]    
        #### mean and std ###
        mean = pred.mean(axis=0)
        std  = pred.std(axis=0)
        true=  np.load("ex_new_1.npy")
        fig, ax = plt.subplots(figsize=(10,8))
        ax.fill_between(extrap_Example[:,4].flatten(), mean-2*std, mean+2*std, color="#fcecca")
        ax.fill_between(extrap_Example[:,4].flatten(), mean-std, mean+std, color="#ffda8f", label= r'$\rm PCNNs$')
        plt.plot(extrap_Example[:,4].flatten(), mean, color= "#fc7600",lw=2)
        plt.plot(extrap_Example[:,4].flatten(),true,'o',color='r', label=r'$\rm True $' )
    
        ax.set_xlabel(r"$\Phi$", fontsize=40)
        ax.text(0.02, 0.3,  r"$\rm X_{bj}=0.365$", transform=ax.transAxes, fontsize = 25)
        ax.text(0.02, 0.2, r"$\rm t= -0.2 ~ GeV^{2}$", transform=ax.transAxes, fontsize = 25)
        ax.text(0.02, 0.1, r"$\rm Q2= 2.0~ GeV^{2}$", transform=ax.transAxes, fontsize = 25)
        plt.legend(fontsize =30, loc= 'upper center')
        ax.set_xticks([0,100,200,300],["0","100","200","300"],fontsize=20)
        ax.set_yticks([0.04,0.05,0.06,0.08],["0.04","","0.06","0.08"],fontsize=20)
        fig.text(-0.06, 0.5, r'$\rm\sigma_{UU}$ \rm(nb/GeV$^4$)',size=40,va='center', rotation='vertical')
        plt.tight_layout()
        plt.show()




    
