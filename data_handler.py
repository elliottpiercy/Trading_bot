import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class data_handler():

    # Initialise variables
    def __init__(self,path,forecast_horizon,input_seq_len,train_val_split,backtest_windows,shuffled,seed):
        self.path = path
        self.forecast_horizon = forecast_horizon
        self.input_seq_len = input_seq_len
        self.train_val_split = train_val_split
        self.backtest_windows = backtest_windows
        self.shuffled = shuffled
        self.seed = seed
        self.scaling_factor = None
        self.scaling_bias = None


    # Scale the data between 0 and 1
    def scale(self,data):
        scaling_factor = np.ptp(data)
        scaling_bias = np.min(data)
        data = (data - scaling_bias)/scaling_factor
        self.scaling_factor = scaling_factor
        self.scaling_bias = scaling_bias
        return data,scaling_factor,scaling_bias


    def rescale(self,data):
        return (data*self.scaling_factor)+self.scaling_bias


    def create_boolean_trend(self,data):
        for i in range(data.shape[0]):
            diff = np.diff(data[i])
            ones = np.where(diff>=0)[0]
            zeros = np.where(diff<0)[0]
            data[i][ones] = 1
            data[i][zeros] = 0
        return data


    # Create sliding window over the data only shuffling the training dataset
    def window(self,data,test=False,shuffled = False):

        # Initial vectors
        X = np.zeros((1,self.input_seq_len))
        Y = np.zeros((1,self.forecast_horizon))




        # Loop over data and create windows
        for loc in range(0,len(data),(self.forecast_horizon)):

            # Try to assign all the data
            try:
                X = np.concatenate((X,np.reshape(data[loc:loc+self.input_seq_len],(1,self.input_seq_len))),axis=0)
                Y = np.concatenate((Y,np.reshape(data[loc+self.input_seq_len:loc+self.input_seq_len+self.forecast_horizon],(1,self.forecast_horizon))),axis=0)
            # Might have afew datapoints left over which cause an exception. Just ignore these points
            except:
                X = np.delete(X,-1,axis=0)
                break


        # Delete vector of 0s
        X = np.delete(X,0,axis=0)
        Y = np.delete(Y,0,axis=0)

        # Shuffle training set
        if not test and self.shuffled:

            # Shuffle
            np.random.seed(self.seed)
            np.random.shuffle(X)
            np.random.seed(self.seed)
            np.random.shuffle(Y)

            # Assign validation set data
            indicies = int(len(X)*(1-self.train_val_split))
            X_v = X[:indicies]
            Y_v = Y[:indicies]

            # Delete validation set data leaving the training data
            X = np.delete(X,indicies,axis=0)
            Y = np.delete(Y,indicies,axis=0)

            return X,Y,X_v,Y_v

        # Data is not shuffled here resulting in a linear trading scheme
        elif not test and not self.shuffled:


            # Assign validation set data
            indicies = int(X.shape[0]*(self.train_val_split))

            X_v = X[:indicies]
            Y_v = Y[:indicies]

            X_t = X_v
            Y_t = Y_v
            # Delete validation set data leaving the training data
            X_v = np.delete(X,np.arange(indicies),axis=0)
            Y_v = np.delete(Y,np.arange(indicies),axis=0)

            return X_t,Y_t,X_v,Y_v
        else:
            return X,Y


    # Split the data into train/val/test. Test set is only used for backtesting and is currently 4 windows worth
    def split(self,data):


        test_cutoff = len(data) - (self.input_seq_len+(self.backtest_windows*self.forecast_horizon))

        test = data[test_cutoff:]
        remainder = data[:test_cutoff]

        X_train,Y_train,X_val,Y_val = self.window(remainder,test = False,shuffled=False)
        X_test,Y_test = self.window(test,test = True,shuffled=False)

        return X_train,Y_train,X_val,Y_val,X_test,Y_test



    def delete_nan(self,d1,d2):

        d1indicies = np.where(np.isnan(d1))[0]
        d2indicies = np.where(np.isnan(d2))[0]


        if np.array_equal(d1indicies,d2indicies):
            d1 = np.delete(d1,d1indicies)
            d2 = np.delete(d2,d2indicies)
        else:
            raise IndexError('Unable to delete NaN values. Index locations are not equal.')

        return d1,d2


    # Read the data from the text file allocated by path.
    def get_data(self,scale = True,daily = False):

        if daily:
            data = np.asarray(pd.read_csv(self.path))
            open_closed_price = np.reshape(np.concatenate((data[:,1],data[:,2]),axis=0),(2,len(data[:,1])))

        else:
            data = np.asarray(pd.read_csv(self.path, index_col=0, header=[0, 1]).sort_index(axis=1))
            d1,d2 = self.delete_nan(data[:,162][1:],data[:,163][1:])
            open_closed_price = np.reshape(np.concatenate((d1,d2),axis=0),(2,len(d1)))




        # Calculate the price by averaging open and close price
        price = np.mean(open_closed_price,axis=0)
        price = price[:int(len(price)/self.forecast_horizon)*self.forecast_horizon]

        plt.plot(price)
        plt.show()

        # Scale, split and window
        price,scaling_factor,scaling_bias = self.scale(price)
        X_train,Y_train,X_val,Y_val,X_test,Y_test = self.split(price)
        return X_train,Y_train,X_val,Y_val,X_test,Y_test,scaling_factor,scaling_bias
