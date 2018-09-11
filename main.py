import pandas as pd
import numpy as np

def step_gradient_descent(data) :
  weigh = np.array(data.weigh)
  height = np.array(data.height)

  learning_rate = 0.00042
  teta_0 = 0
  teta_1 = 0
  N = len(weigh)
  cost_error = []
  iterations = 10000
  
  for i in range(0, iterations):
    height_predicted = (teta_1 * weigh) + teta_0 
    teta_0_d = (1/N)*sum(height_predicted - height) 
    teta_1_d = (1/N)*sum(weigh*(height_predicted - height))
    cost_error = (1/(2*N)) * sum([val**2 for val in (height_predicted - height)])
   
    teta_0 = teta_0 - learning_rate * teta_0_d
    teta_1 = teta_1 - learning_rate * teta_1_d

  return {
    'teta_0': teta_0,
    'teta_1': teta_1,
    'cost_error': cost_error
  }

def predict(data, weigh) : 
  equation = step_gradient_descent(data)
  predicted_height = float( equation['teta_0'] + equation['teta_1'] * weigh )
  return round(predicted_height, 2)


def run() :
  data = pd.read_csv('data.csv')
  weigh = 64
  print("A {}kg person should be {}cm".format(weigh,predict(data, weigh) ))

if __name__ == '__main__':
  run()