import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
import policy_generator as pg

class model():

    def __init__(self,forecast_horizon,backtest_from):
        self.forecast_horizon = forecast_horizon
        self.backtest_from = backtest_from
        self.optimal_params = None
        self.wallet_change = []


    def mse(self,truth,pred):
        return np.mean((truth-pred)**2)


    def get_optimal_parameters(self,data):

        p = d = q = range(0, 2) # Define the p, d and q parameters to take any value between 0 and 2
        order = list(itertools.product(p, d, q)) # Generate all tuples of p, q and q
        seasonal_order = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))] # Generate all different combinations of seasonal p, q and q triplets


        location = int(self.backtest_from*0.75)
        parameter_optimisation_forecast_step = self.backtest_from - location

        X_val = data[:location]
        Y_val = data[location:self.backtest_from]
        min_err = 100000000
        self.optimal_params = order[0]

        for i in order:
            for j in seasonal_order:
                try:
                    mod = sm.tsa.statespace.SARIMAX(np.ndarray.tolist(X_val),
                                                    order=i,
                                                    seasonal_order=j,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()

                                                    # forecast 8 weeks ahead (validation set) and find the optimal parameters
                    forecast = np.asarray(results.get_forecast(steps=parameter_optimisation_forecast_step).predicted_mean)
                    err = self.mse(Y_val,forecast)
                    if err<min_err:
                        min_err = err
                        optimal_params = i,j
                except:
                    continue


        self.optimal_params = optimal_params
        print("Model parameters optimised.")

    def backtest(self,data,wallet):
        print("Backtesting in progress...")

        p = pg.policy_generator(wallet)
        self.wallet_change = [wallet]

        optimal_order,optimal_seasonal_order = self.optimal_params
        for loc in range(self.backtest_from,len(data)-self.forecast_horizon,self.forecast_horizon):

            mod = sm.tsa.statespace.SARIMAX(np.ndarray.tolist(data[:loc]),
                                            order=optimal_order,
                                            seasonal_order=optimal_seasonal_order,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            forecast = np.asarray(results.get_forecast(steps=self.forecast_horizon).predicted_mean)


            p.generate(forecast)
            p.execute_policy(true_data = data[loc:loc+self.forecast_horizon])
            self.wallet_change = np.append(self.wallet_change,p.wallet)

        print("Backtesting complete")
        return p.wallet
