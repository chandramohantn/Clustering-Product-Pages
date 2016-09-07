from keras.layers.core import Dense, Activation, TimeDistributedDense
from keras.models import Model
from keras.layers import LSTM, Input, merge
import numpy as np

def load_data(file_path, n, l, s, item_vec):

	val_path = file_path + 'LSTM_category/' + str(n) + '.npz'
	d = np.load(val_path)
	x = d['data']
	y = d['lab']
	x1 = Process_data(x, l, 15)

	val_path = file_path + 'LSTM_time/' + str(n) + '.npz'
	d = np.load(val_path)
	x = d['data']
	y = d['lab']
	x2 = Process_data(x, l, 194)
	xs = np.concatenate((x1, x2), axis=2)

	val_path = file_path + 'LSTM_item/' + str(n) + '.npz'
	d = np.load(val_path)
	x = d['data']
	y = d['lab']
	x3 = Process_data_item(x, l, 256, item_vec)
	x = np.concatenate((xs, x3), axis=2)

	z = []
	for i in range(len(x)):
		if y[i, 0] == '0':
			z.append([1, 0])
		else:
			z.append([0, 1])
	y = np.asarray(z)
	return x1, x2, x3, x, y

def Process_data(x, l, s):
	a = [0] * s
	z = []
	for i in range(x.shape[0]):
		m = x[i].shape[0]
		y = []
		if (l-m) > 0:
			for j in range(l-m):
				y.append(a)
			for j in x[i]:
				y.append(j)
		else:
			for j in range(l):
				y.append(x[i][j])
		z.append(y)
	z = np.array(z)
	return z

def Process_data_item(x, l, s, item_vec):
	a = [0] * s
	z = []
	for i in range(x.shape[0]):
		m = x[i].shape[0]
		y = []
		if (l-m) > 0:
			for j in range(l-m):
				y.append(a)
			for j in x[i]:
				b = item_vec[j][0].tolist()
				y.append(b)
		else:
			for j in range(l):
				b = item_vec[x[i][j]][0].tolist()
				y.append(b)
		z.append(y)
	z = np.array(z)
	return z

def Train_data(train_path, val_path, limits, epochs, max_len, s, item_vec):
	val_loss = 1000000.0
	for e in range(epochs):
		for l in range(limits):
			[x1, x2, x3, x, y] = load_data(train_path, l, max_len, s, item_vec)
			loss = model.train_on_batch([x1, x2, x3, x], y)
			print('Epoch: ' + str(e) + ' Train batch: ' + str(l) + ' Train Loss: ' + str(loss))
		model.save_weights('model/lstm_moe_filter_buy_w_' + str(e), overwrite=True)

		print('Validating .........')
		loss = 0.0
		for i in range(444):
			[x1, x2, x3, x, y] = load_data(val_path, i, max_len, s, item_vec)
			l = model.evaluate([x1, x2, x3, x], y)
			#print('Val Loss: ' + str(l))
			loss += l
		if loss < val_loss:
			val_loss = loss
			model.save_weights('model/best_lstm_moe_filter_buy_w', overwrite=True)
		print('Val Loss: ' + str(loss))


def Test_model(test_path, limits, max_len, s, item_vec):
	labl = []
	prba = []
	for i in range(limits):
		[x1, x2, x3, x, y] = load_data(test_path, i, max_len, s, item_vec)
		#loss = model.evaluate([x1, x2, x3, x], y)
		#print('Test Loss: ' + str(loss))
		#clas = model.predict_classes([x1, x2, x3, x])
		#prob = model.predict_proba([x1, x2, x3, x])
		#labl.append(clas.tolist())
		#prba.append(prob.tolist())
		prob = model.predict([x1, x2, x3, x])
		prba.append(prob.tolist())
	return prba

max_len = 50
s = 15
a = np.load('../LSTM_category/model/layer_weights.npz')
weights = a['wts']
print('Build expert 1...')
input1 = Input(shape=(max_len, s))
m11 = TimeDistributedDense(128, input_shape=(max_len, s), weights=weights[0])(input1)
m12 = Activation('relu')(m11)
m13 = LSTM(64, return_sequences=True, weights=weights[2])(m12)
m14 = LSTM(64, return_sequences=True)(m13)
m15 = LSTM(64)(m14)
m16 = Dense(2, activation='softmax')(m15)

s = 194
a = np.load('../LSTM_time/model/layer_weights.npz')
weights = a['wts']
print('Build expert 2...')
input2 = Input(shape=(max_len, s))
m21 = TimeDistributedDense(128, input_shape=(max_len, s), weights=weights[0])(input2)
m22 = Activation('relu')(m21)
m23 = LSTM(64, return_sequences=True, weights=weights[2])(m22)
m24 = LSTM(64, return_sequences=True, weights=weights[3])(m23)
m25 = LSTM(64, weights=weights[4])(m24)
m26 = Dense(2, activation='softmax', weights=weights[5])(m25)

s = 256
a = np.load('../LSTM_item/model/layer_weights.npz')
weights = a['wts']
print('Build expert 3...')
input3 = Input(shape=(max_len, s))
m31 = TimeDistributedDense(128, input_shape=(max_len, s), weights=weights[0])(input3)
m32 = Activation('relu')(m31)
m33 = LSTM(64, return_sequences=True, weights=weights[2])(m32)
m34 = LSTM(64, return_sequences=True, weights=weights[3])(m33)
m35 = LSTM(64, weights=weights[4])(m34)
m36 = Dense(2, activation='softmax', weights=weights[5])(m35)

print('Preparing filters ...')
f = merge([m15, m25, m35], mode='concat')

print('Preparing stacker ...')
stacker = merge([m16, m26, m36], mode='concat')

s = 465
print('Build Gating Network ....')
input123 = Input(shape=(max_len, s))
g1 = TimeDistributedDense(256, input_shape=(max_len, s))(input123)
g2 = Activation('relu')(g1)
g3 = LSTM(192)(g2)
g4 = merge([f, g3], mode='mul')
g5 = Dense(6, activation='softmax')(g4)
g6 = merge([stacker, g5], mode='mul')
g7 = Dense(2, activation='softmax')(g6)
model = Model(input=[input1, input2, input3, input123], output=[g7])
model.compile(loss='binary_crossentropy', optimizer='adam')

a = np.load('../../Full_data/Word2Vec_items.npz')
item_vec = a['data']
'''
print("Training...")
train_path = '../../Train_data/Train/New/'
val_path = '../../Train_data/Val/'
Train_data(train_path, val_path, 5253, 10, max_len, s, item_vec)
print('Training complete ......')
'''
print("Testing...")
model.load_weights('model/best_lstm_moe_filter_buy_w')
test_path = '../../Train_data/Test/'
prba = Test_model(test_path, 988, max_len, s, item_vec)
	
f = open('Buys_test_out.csv', 'w');
for i in prba:
	for j in i:
		if j[0] > j[1]:
			f.write('0' + '\n')
		else:
			f.write('1' + '\n')
print('Finished .......')
f.close()

f = open('Buys_test_prob.csv', 'w');
for i in prba:
	for j in i:
		f.write(str(j[0]) + ',' + str(j[1]) + '\n')
print('Finished .......')
f.close()
