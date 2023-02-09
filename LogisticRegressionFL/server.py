import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

def fit_round(server_round):
	return {"server_round":server_round}


def get_eval_fn(model):
	df = pd.read_csv('./dataset/textdatamy.csv')
	y = df["HeartDisease"].to_numpy(int, copy=False)
	x = df.iloc[:,1:-1].values
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

	def eval(server_round, parameters, config):
		model.coef_ = parameters[0]
		if model.fit_intercept:
				model.intercept_ = parameters[1]
		loss = log_loss(y_test, model.predict_proba(x_test))
		accuracy = model.score(x_test, y_test)

		return loss, {"accuracy": accuracy}
	
	return eval



if __name__ == "__main__":
# Create strategy and run server
	model = LogisticRegression()
	n_classes = 2
	n_features = 9
	model.classes_ = np.array([i for i in range(2)])
	model.coef_ = np.zeros((n_classes, n_features))
	if model.fit_intercept:
			model.intercept_ = np.zeros((n_classes,))
	
	strategy = fl.server.strategy.FedAvg(
			min_available_clients=2,
			evaluate_fn=get_eval_fn(model),
			on_fit_config_fn=fit_round,
	)

	# Start Flower server for three rounds of federated learning
	fl.server.start_server(
					server_address = "0.0.0.0:8080", 
					config=fl.server.ServerConfig(num_rounds=3),
					strategy = strategy
	)