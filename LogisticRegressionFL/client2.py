import flwr as fl
import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

local_data = pd.read_csv('./dataset/textdatamy.csv')
local_data = pd.DataFrame(local_data.iloc[150000:, :].values)
y = local_data.iloc[:, 10].values
x = local_data.iloc[:,1:-1].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = LogisticRegression()
model.fit(x_train,y_train)

class FlowerClient(fl.client.NumPyClient):
	def get_parameters(self, config):
		if model.fit_intercept:
			params = [
        model.coef_,
        model.intercept_,
      ]
		else:
			params = [
				model.coef_,
			]
		return params

	def fit(self, parameters, config):
			model.coef_ = parameters[0]
			if model.fit_intercept:
				model.intercept_ = parameters[1]
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				model.fit(x_train, y_train)
			if model.fit_intercept:
				params = [
					model.coef_,
					model.intercept_,
				]
			else:
				params = [
					model.coef_,
				]
			return params, len(x_train), {}

	def evaluate(self, parameters, config):
			model.coef_ = parameters[0]
			if model.fit_intercept:
				model.intercept_ = parameters[1]
			loss = log_loss(y_test, model.predict_proba(x_test))
			accuracy = model.score(x_test, y_test)
			print("Eval accuracy : ", accuracy)
			return loss, len(x_test), {"accuracy": accuracy}

fl.client.start_numpy_client(
	server_address="0.0.0.0:8080",
	client=FlowerClient(),
)