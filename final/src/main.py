from model.rf import RandomForest
from data.data_processing import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import json


def read_json(file):
	with open(file, 'r', encoding='utf_8_sig') as f:
		dic = json.load(f)
	return dic

def main():
	data_file = '../data/comic_data_32.json'
	comic_dic = read_json(data_file)

	dl = DataLoader(comic_dic)
	x, y = dl.create_data()
	features = dl.get_feature_name()
	print(len(x), len(y))
	print(len(x[0]))
	print(len(features))

	# データの分割
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
	_x_train, _x_valid, _y_train, _y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

	# パラメータサーチ
	RF = RandomForest(x_train=_x_train, y_train=_y_train, x_test=_x_valid, y_test=_y_valid)
	params = RF.run(use_optuna=True, n_trials=50)

	# 5 分割交差検証
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
	main()