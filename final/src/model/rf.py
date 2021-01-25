from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import optuna
import matplotlib.pyplot as plt
import numpy as np
import sys, os

class RandomForest:
	def __init__(self, x_train, y_train, x_test, y_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test

	def create_model(self, n_estimators=10, 
							max_depth=30, 
							min_samples_split=2, 
							max_features=1
							):

		model = RFC(n_estimators=n_estimators, 
					max_depth=max_depth, 
					min_samples_split=min_samples_split,
					max_features=max_features,
					random_state=0, 
					verbose=0)

		self.model = model

		return model

	def run(self, use_optuna=False, n_trials=100, 
					n_estimators=10, 
					max_depth=30, 
					min_samples_split=2, 
					max_features=1
					):

		if use_optuna:
			params = self.search_param(n_trials=n_trials)
			return params

		else:
			model = self.create_model(n_estimators=n_estimators, 
									max_depth=max_depth, 
									min_samples_split=min_samples_split, 
									max_features=max_features
									)
			self.fit(model)
			score = self.test(model=model, x_test=self.x_test, y_test=self.y_test)
			fti = model.feature_importances_
			return score, fti
			

	def fit(self, model=None):
		if model == None:
			model = self.model

		model.fit(self.x_train, self.y_train)

		train_score = model.score(self.x_train, self.y_train)
		test_score = model.score(self.x_test, self.y_test)

		print('train score: ', train_score)
		print('test score: ', test_score)

		return train_score, test_score

	def test(self, model=None, x_test=[], y_test=[]):
		if model==None:
			model = self.model

		score = model.score(x_test, y_test)
		print('score: ', score)

		return score


	def search_param(self, n_trials=100):

		def objective(trial):
			n_estimators = trial.suggest_categorical('n_estimators',  [2, 4, 8, 16, 32, 64, 128, 256])
			max_depth = trial.suggest_categorical('max_depth', [2, 4, 8, 16, 32, 64, 128, 256])
			min_samples_split = trial.suggest_categorical('min_samples_split', [2, 0.1, 0.5, 1.0])
			max_features = trial.suggest_int('max_features', 1, len(self.x_train[0]))

			model = self.create_model(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)
			train_score, test_score = self.fit(model=model)

			return accuracy_score(self.y_test, model.predict(self.x_test))

		study = optuna.create_study(direction='maximize')
		study.optimize(objective, n_trials=n_trials)

		# 結果の表示
		print("  Number of finished trials: ", len(study.trials))
		print("Best trial:")
		trial = study.best_trial
		print("  Value: ", trial.value)
		print("  Params: ")
		params = {}
		for key, value in trial.params.items():
			print(f"    {key}: {value}")
			params[key] = value

		return params

def test():
	print('test rf')
	# データ
	iris = datasets.load_iris()
	x = iris.data 
	y = iris.target

	print(x.shape)
	print(y.shape)

	# データ前処理
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



	RF = RandomForest(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
	params = RF.run(use_optuna=True, n_trials=10)

	kf = StratifiedKFold(n_splits=5, shuffle=True)
	accracy = []
	ftis = []
	i = 0
	for train_index, test_index in kf.split(x, y):
		print("######################")
		print(i, " 回目")
		X_train, X_test = x[train_index], x[test_index]
		Y_train, Y_test = y[train_index], y[test_index]
		_RF = RandomForest(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)
		score, fti = _RF.run(use_optuna=False, 
						n_estimators=params['n_estimators'], 
						max_depth=params['max_depth'], 
						min_samples_split=params['min_samples_split'], 
						max_features=params['max_features']
						)
		print(fti)
		ftis.append(fti)
		accracy.append(score)
		i += 1
	print('########## result ############')
	print('score')
	print('mean: ', np.mean(accracy))
	print('std: ', np.std(accracy))
	print('fti')
	fti_means = np.mean(ftis, axis=0)
	fti_stds = np.std(ftis, axis=0)
	for i in range(len(fti)):
		print(i, fti_means[i], fti_stds[i])

if __name__ == '__main__':
	test()