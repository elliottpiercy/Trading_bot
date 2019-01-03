# Policy generator. Generates the trading policy based on the forecast. Currently buys low, sells high

import numpy as np


class policy_generator():

    # Initialise variables.
    def __init__(self,wallet):
        self.policy = None
        self.stocks_owned = 0
        self.wallet = wallet
        self.change = [wallet]

    # Positive number check
    def is_positive(self,x):
        if x>=0:
            return True
        else:
            return False

    # Find change of each value wrt the next
    def diff_seq(self,data):
        diff = []
        for i in range(len(data)-1):
            diff = np.append(diff,data[i+1]-data[i])
        return diff


    # Generate the forecast
    def generate(self,forecast):

        diff = self.diff_seq(forecast)
        policy = []

        # For the first value, if low buy otherwise wait
        if self.is_positive(diff[0]):
            policy = ['buy']
        else:
            policy = ['wait']

        # Slide window of size 3 over the diff array looking for high and low points
        for i in range(2,len(diff)+1):
            window = np.asarray(forecast[i-2:i+1])

            # high point - sell high
            if window[1]> window[0] and window[1]>window[2]:
                 policy = np.append(policy,'sell')
            # low point - buy low
            elif window[1]< window[0] and window[1]<window[2]:
                 policy = np.append(policy,'buy')
            # Otherwise wait for optimal point
            else:
                 policy = np.append(policy,'wait')

        # Finally sell at the end of the sequence
        policy = np.append(policy,'sell')
        self.policy = policy
        return policy


    # Buy stocks -- buy as many as possible
    def buy_stocks(self,price):
        self.stocks_owned = int(self.wallet/price)
        self.wallet = self.wallet - (self.stocks_owned*price)


    # Sell stocks -- sell all stocks at market rate
    def sell_stocks(self,price):
        self.wallet = self.wallet + (self.stocks_owned*price)
        self.stocks_owned = 0


    # Iterate over the policy and alter wallet accordingly based on market rate
    def execute_policy(self,true_data):
        for i in range(len(self.policy)):
            if self.policy[i] == 'buy':
                self.buy_stocks(true_data[i])
            elif self.policy[i] =='wait':
                continue
            elif self.policy[i] == 'sell':
                self.sell_stocks(true_data[i])
            self.change = np.append(self.change,self.wallet)
