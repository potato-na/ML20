from operator import itemgetter
from copy import deepcopy
import io
import sys
import os
from tqdm import tqdm
import glob
import pathlib
from datetime import datetime
from pprint import pprint
import json

# データセットの作成
class DataLoader:
	def __init__(self, comic_dic):
		self.comic_dic = comic_dic
		self.dp = DataProcessing()

	def create_data(self):
		x = []
		y = []

		for title in tqdm(self.comic_dic):
			for page in self.comic_dic[title]:
				# 1 ページ分の素性を獲得
				self.dp.set_belonging_frame_to_text(self.comic_dic[title][str(page)])
				list_feature_vecs = self.dp.make_feature_vec(self.comic_dic[title][str(page)], create_for_serif=True)

				for feature_vec in list_feature_vecs:
					# 個々の特徴量を獲得
					x.append(feature_vec['feature_vec'])
					y.append(feature_vec['label'])

		return x, y

	def get_feature_name(self):
		feature_names = ['text_coordinate_difference_xmin', 'text_coordinate_difference_ymin', 'text_coordinate_difference_xmax', 'text_coordinate_difference_ymax', 
				'text_0_absolute_coodinate_xmin', 'text_0_absolute_coodinate_ymin', 'text_0_absolute_coodinate_xmax', 'text_0_absolute_coodinate_ymax', 
				'text_0_is_4scene', 
				'text_1_absolute_coodinate_xmin', 'text_1_absolute_coodinate_ymin', 'text_1_absolute_coodinate_xmax', 'text_1_absolute_coodinate_ymax', 
				'text_1_is_4scene', 
				'frame_coordinate_difference_xmin', 'frame_coordinate_difference_ymin', 'frame_coordinate_difference_xmax', 'frame_coordinate_difference_ymax',  
				'frame_0_absolute_coodinate_xmin', 'frame_0_absolute_coodinate_ymin', 'frame_0_absolute_coodinate_xmax', 'frame_0_absolute_coodinate_ymax',  
				'frame_1_absolute_coodinate_xmin', 'frame_1_absolute_coodinate_ymin', 'frame_1_absolute_coodinate_xmax', 'frame_1_absolute_coodinate_ymax'
				]
		return feature_names



# 学習データ作成などの前処理をまとめたクラス
class DataProcessing:
	def __init__(self):
		pass

	# 台詞の所属コマを調べる　入力: 1ページ分のデータ, 効果: textに所属コマ(belonging_frame)追加
	def set_belonging_frame_to_text(self, one_page_data):

		for serif_id in one_page_data['text']:
			xmin_s = int(one_page_data['text'][serif_id]['xmin'])
			ymin_s = int(one_page_data['text'][serif_id]['ymin'])
			xmax_s = int(one_page_data['text'][serif_id]['xmax'])
			ymax_s = int(one_page_data['text'][serif_id]['ymax'])

			belong_frame_id = 'None'
			max_vertex = 0
			min_g_distance = 1000000000
			min_priority_distance = 100000000

			gx_s = (xmin_s + xmax_s) / 2.0
			gy_s = (ymax_s + ymax_s) / 2.0

			# コマに内包される頂点数調べ
			cnt = 0
			for frame_id in one_page_data['frame']:
				xmin_f = int(one_page_data['frame'][frame_id]['xmin'])
				ymin_f = int(one_page_data['frame'][frame_id]['ymin'])
				xmax_f = int(one_page_data['frame'][frame_id]['xmax'])
				ymax_f = int(one_page_data['frame'][frame_id]['ymax'])

				inclusion_vertex = 0

				# コマの左上
				if (xmin_s >= xmin_f) and (ymin_s >= ymin_f) and (xmin_s <= xmax_f) and (ymin_s <= ymax_f):
					inclusion_vertex += 1
				# コマの左下
				if (xmin_s >= xmin_f) and (ymax_s >= ymin_f) and (xmin_s <= xmax_f) and (ymax_s <= ymax_f):
					inclusion_vertex += 1
				# コマの右上
				if (xmax_s >= xmin_f) and (ymin_s >= ymin_f) and (xmax_s <= xmax_f) and (ymin_s <= ymax_f):
					inclusion_vertex += 1
				# コマの右下
				if (xmax_s >= xmin_f) and (ymax_s >= ymin_f) and (xmax_s <= xmax_f) and (ymax_s <= ymax_f):
					inclusion_vertex += 1

				gx_f = (xmin_f + xmax_f) / 2.0
				gy_f = (ymin_f + ymax_f) / 2.0

				g_distance = ((gx_s-gx_f)**2 + (gy_s-gy_f)**2) ** (1/2)

				# コマの四辺と台詞重心の距離平均
				edge_distance = (abs(gx_s-float(xmin_f)) + abs(gx_s-float(xmax_f)) + abs(gy_s-float(ymin_f)) + abs(gy_s-float(ymax_f))) / 4


				# 台詞の四隅が全てコマの内部にあり、コマと台詞の重心距離が優先最小距離より小さければ
				if inclusion_vertex == 4 and edge_distance < min_priority_distance:
					belong_frame_id = frame_id
					min_priority_distance = edge_distance
					max_vertex = inclusion_vertex
					cnt = 1

				# 含有頂点数が最大頂点数と同じ　かつ　コマと台詞の重心距離が最小距離より小さければ	
				if inclusion_vertex == max_vertex and g_distance < min_g_distance and inclusion_vertex < 4 and not inclusion_vertex == 0:
					belong_frame_id = frame_id
					min_g_distance = g_distance

				# 含有頂点数が最大頂点数より多ければ
				if inclusion_vertex > max_vertex:
					belong_frame_id = frame_id
					max_vertex = inclusion_vertex
					min_g_distance = g_distance
					cnt = 1


			one_page_data['text'][serif_id]['belonging_frame'] = belong_frame_id

	# 素性ベクトルを作る 入力: １ページ内のデータ, 出力: [{forward_serif_id:0, backward_serif_id:0, feature_vec:0, label:0}]
	def make_feature_vec(self, one_page_data, create_for_serif=True):
		isPosi = True
		labels = []
		list_feature_vecs= []

		order_sorted_list = []

		if create_for_serif:

			for serif_id in one_page_data['text']:
				if one_page_data['text'][serif_id]['annotation_order'] == 'None':
					continue

				annotation_order_s = int(one_page_data['text'][serif_id]['annotation_order'])
				xmin_s = int(one_page_data['text'][serif_id]['xmin'])
				ymin_s = int(one_page_data['text'][serif_id]['ymin'])
				xmax_s = int(one_page_data['text'][serif_id]['xmax'])
				ymax_s = int(one_page_data['text'][serif_id]['ymax'])
				is4panel_s = int(one_page_data['text'][serif_id]['is4panel'])
				order_sorted_list.append((annotation_order_s, serif_id, xmin_s, ymin_s, xmax_s, ymax_s, is4panel_s))
		else:
			# frame用
			for frame_id in one_page_data['frame']:

				annotation_order_f = int(one_page_data['frame'][frame_id]['annotation_order'])
				xmin_f = int(one_page_data['frame'][frame_id]['xmin'])
				ymin_f = int(one_page_data['frame'][frame_id]['ymin'])
				xmax_f = int(one_page_data['frame'][frame_id]['xmax'])
				ymax_f = int(one_page_data['frame'][frame_id]['ymax'])
				is4panel_f = int(one_page_data['frame'][frame_id]['is4panel'])
				order_sorted_list.append((annotation_order_f, frame_id, xmin_f, ymin_f, xmax_f, ymax_f, is4panel_f))


		order_sorted_list.sort()


		for i in range(len(order_sorted_list)):
			for j in range(i+1, len(order_sorted_list)):
				if isPosi: #正順
					k = i
					l = j
					labels.append(0)
				else:		#逆順
					k = j
					l = i
					labels.append(1)

				feature_vec = [o-t for o, t in zip(order_sorted_list[k][2:6], order_sorted_list[l][2:6])] # 台詞の座標差分
				feature_vec.extend(order_sorted_list[k][2:6]) # 先台詞/コマの座標
				feature_vec.append(order_sorted_list[k][6]) # 先台詞/コマの４コマフラグ
				feature_vec.extend(order_sorted_list[l][2:6]) # 後台詞/コマの座標
				feature_vec.append(order_sorted_list[l][6]) # 後台詞/コマの４コマフラグ

				if create_for_serif:
					# 所属コマの処理

					k_frame_id = one_page_data['text'][order_sorted_list[k][1]]['belonging_frame'] # 先台詞の所属コマ
					l_frame_id = one_page_data['text'][order_sorted_list[l][1]]['belonging_frame'] # 後台詞の所属コマ

					if k_frame_id == 'None':
						# 所属コマNoneの場合, そのテキスト自身の領域を所属コマの座標として設定
						_xmin = order_sorted_list[k][2]
						_ymin = order_sorted_list[k][3]
						_xmax = order_sorted_list[k][4]
						_ymax = order_sorted_list[k][5]
						list_f_k = (_xmin, _ymin, _xmax, _ymax)

					else:
						xmin_f_k = int(one_page_data['frame'][k_frame_id]['xmin'])
						ymin_f_k = int(one_page_data['frame'][k_frame_id]['ymin'])
						xmax_f_k = int(one_page_data['frame'][k_frame_id]['xmax'])
						ymax_f_k = int(one_page_data['frame'][k_frame_id]['ymax'])
						list_f_k = (xmin_f_k, ymin_f_k, xmax_f_k, ymax_f_k)

					if l_frame_id == 'None':
						_xmin = order_sorted_list[l][2]
						_ymin = order_sorted_list[l][3]
						_xmax = order_sorted_list[l][4]
						_ymax = order_sorted_list[l][5]
						list_f_l = (_xmin, _ymin, _xmax, _ymax)
						
					else:
						xmin_f_l = int(one_page_data['frame'][l_frame_id]['xmin'])
						ymin_f_l = int(one_page_data['frame'][l_frame_id]['ymin'])
						xmax_f_l = int(one_page_data['frame'][l_frame_id]['xmax'])
						ymax_f_l = int(one_page_data['frame'][l_frame_id]['ymax'])
						list_f_l = (xmin_f_l, ymin_f_l, xmax_f_l, ymax_f_l)

					feature_vec.extend([o-t for o, t in zip(list_f_k[:], list_f_l[:])]) # コマの座標差分
					feature_vec.extend(list_f_k[:])
					feature_vec.extend(list_f_l[:])

					dic = {}
					dic['forward_serif_id'] = order_sorted_list[k][1]
					dic['backward_serif_id'] = order_sorted_list[l][1]
					dic['feature_vec'] = feature_vec
					dic['label'] = labels[-1]
					list_feature_vecs.append(dic)

				else:
					# frame用
					dic = {}
					dic['forward_frame_id'] = order_sorted_list[k][1]
					dic['backward_frame_id'] = order_sorted_list[l][1]
					dic['feature_vec'] = feature_vec
					dic['label'] = labels[-1]
					list_feature_vecs.append(dic)

				isPosi = not isPosi


		return list_feature_vecs