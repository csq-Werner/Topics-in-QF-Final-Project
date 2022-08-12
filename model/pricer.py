import tensorflow as tf
from tensorflow.keras.layers import *
import os
import numpy as np
import scipy
import sobol_seq
import json
import matplotlib.pyplot as plt
import time
import pickle
from tqdm import tqdm
from model.process import *
from model.option import option_eu, option_am
data_dir = './'


"""## Option solver"""

class Pricer:
	def __init__(self, model, cp, exercise, dtype = 'float32'):
		self.model = model ## 'VG','CGMY','NIG','Kou','Merton'
		self.cp = cp ## 'call','put'
		self.exercise = exercise ## 'European', 'American'
		self.K = 100
		self.data_range_fun()
		self.dtype = dtype
		if self.cp not in ['call','put']: raise ValueError
		if self.exercise not in ['European', 'American']: raise ValueError
	
	## specify the range of the random samples
	def data_range_fun(self):
		dic = {}
		K = self.K
		dic['x'] = {'train':{'low': np.log(K/50), 'high':np.log(K*50)},
								'test':{'low': np.log(K/2), 'high':np.log(K*2)},
								'bound':{'low': np.log(K/50), 'high':np.log(K*50)}} 
								
		if self.model in ['CGMY', 'NIG']: ### special
			dic['x']['train']['high'] = np.log(K*500)
			dic['x']['bound']['high'] = np.log(K*500)

		dic['others'] = {'t':{'low': 0, 'high': 3},
										'r':{'low': 0, 'high': 0.1},
										'q':{'low': 0, 'high': 0.1}}

		if self.model == 'VG':
			dic['others'].update({'sig':{'low': 0.1, 'high': 0.5},
															'nu':{'low': 0.1, 'high': 0.6},
															'theta':{'low': -0.5, 'high': -0.1}})
			model_paras = ['sig', 'nu', 'theta']

		elif self.model == 'CGMY':
			dic['others'].update({'sig':{'low': 0.1, 'high': 0.5},
															'nu':{'low': 0.1, 'high': 0.6},
															'theta':{'low': -0.5, 'high': -0.1},
															'Y':{'low': 0, 'high': 1}})
			model_paras = ['sig', 'nu', 'theta', 'Y']

		elif self.model == 'NIG':
			dic['others'].update({'alpha':{'low': 5, 'high': 20},
															'beta':{'low': -2/3, 'high': 2/3}, ### special: beta = beta * alpha
															'delta':{'low': 0.1, 'high': 3}})
			model_paras = ['alpha', 'beta', 'delta']

		elif self.model == 'Merton':
			dic['others'].update({'sig':{'low': 0.1, 'high': 0.5},
															'lam':{'low': 0, 'high': 1},
															'alpha':{'low': -0.5, 'high': 0.5},
															'delta':{'low': 0.01, 'high': 0.5}})
			model_paras = ['sig', 'lam', 'alpha', 'delta']

		elif self.model == 'Kou':
			dic['others'].update({'sig':{'low': 0.1, 'high': 0.5},
															'lam':{'low': 0, 'high': 2},
															'p':{'low': 0, 'high': 1},
															'eta1':{'low': 3, 'high': 15},
														'eta2':{'low': 3, 'high': 15}})
			model_paras = ['sig', 'lam', 'p', 'eta1', 'eta2']
		else:
			raise Exception

		self.data_range = dic
		self.model_paras = model_paras
		self.dim = len(dic['others']) + 1
	
	## draw quasi random samples
	def data_sampler(self):
		quasi_rn = sobol_seq.i4_sobol_generate(self.dim, self.train_size + self.test_size)
		train_rn = quasi_rn[:self.train_size]
		test_rn = quasi_rn[self.train_size:] 

		train_data = {}
		_range = self.data_range['x']['train']
		train_data['x'] = train_rn[:,0,np.newaxis] * (_range['high'] - _range['low']) + _range['low']  
		for i, (key, _range) in enumerate(self.data_range['others'].items()):
			train_data[key] = train_rn[:,i+1,np.newaxis] * (_range['high'] - _range['low']) + _range['low'] 

		test_data = {}
		_range = self.data_range['x']['test']
		test_data['x'] = test_rn[:,0,np.newaxis] * (_range['high'] - _range['low']) + _range['low']  
		train_data['x'][::2] = train_rn[::2,0,np.newaxis] * (_range['high'] - _range['low']) + _range['low'] 
		for i, (key, _range) in enumerate(self.data_range['others'].items()):
			test_data[key] = test_rn[:,i+1,np.newaxis] * (_range['high'] - _range['low']) + _range['low'] 

		self.train_data = train_data
		self.test_data = test_data

	## pre-calculation
	def data_augment(self, data, tag):     

		if self.model == 'VG':
			lambda_p = cgmy_lambda_p_fun(data['sig'], data['nu'], data['theta'])
			lambda_n = cgmy_lambda_n_fun(data['sig'], data['nu'], data['theta'])

			data['ratio_p'] = 1 / lambda_p
			data['ratio_n'] = 1 / lambda_n
			
			## load if calculated
			if os.path.exists(data_dir + 'data/vg_'+tag+'_omega.pkl') and os.path.exists(data_dir + 'data/vg_'+tag+'_sig2.pkl'):
				with open(data_dir + 'data/vg_'+tag+'_omega.pkl', 'rb') as handle:
						data['omega'] = pickle.load(handle)
				with open(data_dir + 'data/vg_'+tag+'_sig2.pkl', 'rb') as handle:
						data['sig2'] = pickle.load(handle) 
			else:
				data['omega'] = np.array([cgmy_omg_eps_fun(sig, nu, theta, 0, self.eps*r_p, self.eps*r_n) 
					for (sig, nu, theta, r_p, r_n) in tqdm(zip(data['sig'], data['nu'], data['theta'], data['ratio_p'], data['ratio_n']))])
				data['sig2'] = np.array([cgmy_sig2_eps_fun(sig, nu, theta, 0, self.eps*r_p, self.eps*r_n) 
					for (sig, nu, theta, r_p, r_n) in tqdm(zip(data['sig'], data['nu'], data['theta'], data['ratio_p'], data['ratio_n']))])
				if not os.path.exists(data_dir + 'data/'):
					os.makedirs(data_dir + 'data/')
										
				with open(data_dir + 'data/vg_'+tag+'_omega.pkl', 'wb') as handle:
						pickle.dump(data['omega'], handle)
				with open(data_dir + 'data/vg_'+tag+'_sig2.pkl', 'wb') as handle:
						pickle.dump(data['sig2'], handle) 

			data['k_n'] = cgmy_k_fun(lambda_n,data['nu'],0,self.y_grid*data['ratio_n'])
			data['k_p'] = cgmy_k_fun(lambda_p,data['nu'],0,self.y_grid*data['ratio_p'])

		elif self.model == 'CGMY':  
			lambda_p = cgmy_lambda_p_fun(data['sig'], data['nu'], data['theta'])
			lambda_n = cgmy_lambda_n_fun(data['sig'], data['nu'], data['theta'])
			
			data['ratio_p'] = 1 / lambda_p
			data['ratio_n'] = 1 / lambda_n
			
			## load if calculated
			if os.path.exists(data_dir + 'data/cgmy_'+tag+'_omega.pkl') and os.path.exists(data_dir + 'data/cgmy_'+tag+'_sig2.pkl'):
				with open(data_dir + 'data/cgmy_'+tag+'_omega.pkl', 'rb') as handle:
						data['omega'] = pickle.load(handle)
				with open(data_dir + 'data/cgmy_'+tag+'_sig2.pkl', 'rb') as handle:
						data['sig2'] = pickle.load(handle) 
			else:
				data['omega'] = np.array([cgmy_omg_eps_fun(sig, nu, theta, Y, self.eps*r_p, self.eps*r_n) 
					for (sig, nu, theta, Y, r_p, r_n) in tqdm(zip(data['sig'], data['nu'], data['theta'], data['Y'], data['ratio_p'], data['ratio_n']))])
				data['sig2'] = np.array([cgmy_sig2_eps_fun(sig, nu, theta, Y, self.eps*r_p, self.eps*r_n) 
					for (sig, nu, theta, Y, r_p, r_n) in tqdm(zip(data['sig'], data['nu'], data['theta'], data['Y'], data['ratio_p'], data['ratio_n']))])
				if not os.path.exists(data_dir + 'data/'):
					os.makedirs(data_dir + 'data/')
				with open(data_dir + 'data/cgmy_'+tag+'_omega.pkl', 'wb') as handle:
						pickle.dump(data['omega'], handle)
				with open(data_dir + 'data/cgmy_'+tag+'_sig2.pkl', 'wb') as handle:
						pickle.dump(data['sig2'], handle)

			data['k_n'] = cgmy_k_fun(lambda_n,data['nu'],data['Y'],self.y_grid*data['ratio_n'])
			data['k_p'] = cgmy_k_fun(lambda_p,data['nu'],data['Y'],self.y_grid*data['ratio_p'])
		
		elif self.model == 'NIG':      
			data['beta'] = data['beta']*(data['alpha']) ### special

			data['ratio_p'] = 1 / data['alpha']
			data['ratio_n'] = 1 / data['alpha']
			
			## load if calculated
			if os.path.exists(data_dir + 'data/nig_'+tag+'_omega.pkl') and os.path.exists(data_dir + 'data/nig_'+tag+'_sig2.pkl'):
				with open(data_dir + 'data/nig_'+tag+'_omega.pkl', 'rb') as handle:
						data['omega'] = pickle.load(handle)
				with open(data_dir + 'data/nig_'+tag+'_sig2.pkl', 'rb') as handle:
						data['sig2'] = pickle.load(handle) 
			else:
				data['omega'] = np.array([nig_omg_eps_fun(alpha, beta, delta, self.eps*r_p, self.eps*r_n) 
					for (alpha, beta, delta, r_p, r_n) in tqdm(zip(data['alpha'], data['beta'], data['delta'], data['ratio_p'], data['ratio_n']))])
				data['sig2'] = np.array([nig_sig2_eps_fun(alpha, beta, delta, self.eps*r_p, self.eps*r_n) 
					for (alpha, beta, delta, r_p, r_n) in tqdm(zip(data['alpha'], data['beta'], data['delta'], data['ratio_p'], data['ratio_n']))])
				if not os.path.exists(data_dir + 'data/'):
					os.makedirs(data_dir + 'data/')
				with open(data_dir + 'data/nig_'+tag+'_omega.pkl', 'wb') as handle:
						pickle.dump(data['omega'], handle)
				with open(data_dir + 'data/nig_'+tag+'_sig2.pkl', 'wb') as handle:
						pickle.dump(data['sig2'], handle)

			data['k_n'] = nig_k_fun(data['alpha'],-data['beta'],data['delta'],self.y_grid*data['ratio_n'])
			data['k_p'] = nig_k_fun(data['alpha'],data['beta'],data['delta'],self.y_grid*data['ratio_p'])

		elif self.model == 'Merton':
			data['sig2'] = merton_sig2_eps_fun(data['sig'])
			data['omega'] = merton_omg_eps_fun(data['lam'], data['alpha'], data['delta'])

			data['ratio_p'] = data['delta']
			data['ratio_n'] = data['delta']
			 
			data['k_n'] = merton_k_fun(data['lam'], data['alpha'], data['delta'], data['alpha']-data['ratio_n']*self.y_grid)
			data['k_p'] = merton_k_fun(data['lam'], data['alpha'], data['delta'], data['alpha']+data['ratio_p']*self.y_grid)

		elif self.model == 'Kou':
			data['sig2'] = kou_sig2_eps_fun(data['sig'])
			data['omega'] =  kou_omg_eps_fun(data['lam'], data['p'], data['eta1'], data['eta2']) 

			data['ratio_p'] = 1 / data['eta1']
			data['ratio_n'] = 1 / data['eta2']
			
			data['k_n'] = kou_k_fun(data['lam'], 1-data['p'], data['eta2'], self.y_grid*data['ratio_n'])
			data['k_p'] = kou_k_fun(data['lam'], data['p'], data['eta1'], self.y_grid*data['ratio_p'])

		else:
			raise Exception

		paras = []
		for key in self.model_paras:
			paras.append(data[key])
			data['paras'] = np.hstack(paras)    
	
	## prepare all data before training
	def data_preparer(self, train_size=500000, test_size=10000):
		self.train_size = train_size
		self.test_size = test_size
		self.data_sampler()

		## z_j in the paper appendix
		if self.model in ['VG']:
			self.eps = 0.02
			y_grid = np.hstack((np.linspace(0.02,5,40, endpoint= False), 
													np.linspace(5,15,30, endpoint= False),
													np.linspace(15,30,21, endpoint= True)))
			
		elif self.model in ['CGMY']:
			self.eps = 0.01     
			y_grid = np.hstack((np.linspace(0.01,0.1,80, endpoint= False),
													np.linspace(0.1,0.5,40, endpoint= False),
													np.linspace(0.5,2.5,40, endpoint= False), 
													np.linspace(2.5,5,10, endpoint= False), 
													np.linspace(5,15,30, endpoint= False),
													np.linspace(15,30,21, endpoint= True)))    

		elif self.model in ['NIG']:
			self.eps = 0.05     
			y_grid = np.hstack((np.linspace(0.05,0.1,10, endpoint= False),
													np.linspace(0.1,0.2,10, endpoint= False),
													np.linspace(0.2,0.4,10, endpoint= False),
													np.linspace(0.4,1,14, endpoint= False),
													np.linspace(1,2.5,14, endpoint= False),
													np.linspace(2.5,5,10, endpoint= False),
													np.linspace(5,10,10,endpoint= False),
													np.linspace(10,20,12,endpoint= False),
													np.linspace(20,40,10,endpoint= False),
													np.linspace(40,80,11,endpoint= True)))

		elif self.model == 'Kou': 
			y_grid = np.hstack((np.linspace(0,5,50, endpoint= False),
													np.linspace(5,10,14, endpoint= False),
													np.linspace(10,20,16, endpoint= False),
													np.linspace(20,30,5, endpoint= True)))

		elif self.model == 'Merton': 
			y_grid = np.hstack((np.linspace(0, 5, 70, endpoint= False),
													np.linspace(5, 7, 21, endpoint= True)))

		else: raise Exception

		weight_grid = np.hstack((np.diff(y_grid)/3,[0])) + np.hstack(([0], np.diff(y_grid)/3))
		weight_grid[1::2] = weight_grid[1::2]*2
		self.y_grid = y_grid
		self.weight_grid = weight_grid

		self.data_augment(self.train_data, 'train')
		self.data_augment(self.test_data, 'test')
		self.train_tensor = {}
		self.test_tensor = {}
		for k,v in self.train_data.items():
			self.train_tensor[k] = tf.convert_to_tensor(v, dtype=self.dtype)
		for k,v in self.test_data.items():
			self.test_tensor[k] = tf.convert_to_tensor(v, dtype=self.dtype)
		self.y_grid = tf.convert_to_tensor(self.y_grid, dtype=self.dtype)
		self.weight_grid = tf.convert_to_tensor(self.weight_grid, dtype=self.dtype)

	## neural network
	def net_builder(self, layers1, layers2, numbers, initial = 'he_normal', batch_normal = False, drop_out = 0):
		
		x_layer = Input(shape = (1), dtype=self.dtype)
		t_layer = Input(shape = (1), dtype=self.dtype)
		r_layer = Input(shape = (1), dtype=self.dtype)
		q_layer = Input(shape = (1), dtype=self.dtype)
		paras_layer = Input(shape = (len(self.model_paras)), dtype=self.dtype)
		mid = concatenate([x_layer, t_layer, r_layer, q_layer, paras_layer], axis = -1)
		y_layer = 0

		for i in range(layers1):
			y_layer += Dense(1, dtype=self.dtype, kernel_initializer=initial)(mid)
			mid = Dense(numbers, dtype=self.dtype, kernel_initializer=initial)(mid)
			if batch_normal:
					mid = BatchNormalization(dtype=self.dtype)(mid)
			if drop_out:
					mid = Dropout(drop_out)(mid)
			mid = tf.nn.silu(mid)

		## singular terms
		mul = Dense(1, dtype=self.dtype, kernel_initializer=initial)(mid) + self.K/10
		mul = tf.nn.softplus(mul) + 1e-6
		bias = Dense(1, dtype=self.dtype, kernel_initializer=initial)(mid) 
		euro_mul_s = tf.exp(-q_layer*t_layer) if self.exercise == 'European' else 1
		euro_mul_k = tf.exp(-r_layer*t_layer) if self.exercise == 'European' else 1
		scaled = (tf.exp(x_layer)*euro_mul_s - self.K*euro_mul_k + bias*t_layer)/(tf.sqrt(t_layer+1e-8))/mul
		if self.cp == 'put':
			scaled = -scaled 
		singu = tf.nn.softplus(scaled) * (tf.sqrt(t_layer+1e-8)) * mul

		mul2 = Dense(1, dtype=self.dtype, kernel_initializer=initial)(mid)+self.K/20
		mul2 = tf.nn.softplus(mul2) + 1e-6
		bias2 = Dense(1, dtype=self.dtype, kernel_initializer=initial)(mid)*self.K/10 
		scaled2 = (tf.exp(x_layer)*euro_mul_s - self.K*euro_mul_k + bias2*t_layer)/(tf.sqrt(t_layer+1e-8))/mul2
		if self.cp == 'put':
			scaled2 = -scaled2 
		singu2 = tf.nn.silu(scaled2) * (tf.sqrt(t_layer+1e-8)) * mul2

		mid = concatenate([mid, singu, singu2], axis = -1)
	
		for i in range(layers2):
				y_layer += Dense(1, dtype=self.dtype, kernel_initializer=initial)(mid)
				mid = Dense(numbers, dtype=self.dtype, kernel_initializer=initial)(mid)
				if batch_normal:
						mid = BatchNormalization(dtype=self.dtype)(mid)
				if drop_out:
						mid = Dropout(drop_out)(mid)
				mid = tf.nn.silu(mid)

		y_layer += Dense(1, dtype=self.dtype, kernel_initializer=initial)(mid)

		self.net = tf.keras.Model([x_layer, t_layer, r_layer, q_layer, paras_layer], y_layer)
		self.net.summary() 

	## functions of saving and loading
	def set_name(self, name):
		self.name = name
	def save_model(self, train_loss_list, val_loss_list):
		opt=tf.keras.optimizers.Adam()
		checkpoint_directory = os.path.join(data_dir, self.name)
		checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
		if not os.path.exists(checkpoint_directory):
				os.makedirs(checkpoint_directory)
		checkpoint = tf.train.Checkpoint(optimizer=opt, model=self.net)
		checkpoint.save(file_prefix=checkpoint_prefix)  
		with open(os.path.join(checkpoint_directory, "loss_list.json"), "w") as f:
			loss_json = {
				"train_loss": train_loss_list,
				"val_loss": val_loss_list
			}
			json.dump(loss_json, f)
	def load_model(self):
		opt=tf.keras.optimizers.Adam()
		checkpoint_directory = os.path.join(data_dir, self.name)
		checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
		if not os.path.exists(checkpoint_directory):
				os.makedirs(checkpoint_directory)
		checkpoint = tf.train.Checkpoint(optimizer=opt, model=self.net)
		status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
	
	## benchmark
	def benchmark(self, S, T, r, q, **paras):
		paras = {k:paras[k] for k in self.model_paras}
		if self.exercise == 'European':
				price = option_eu(self.model, self.cp,S,self.K,T,r,q,**paras)
		elif self.exercise == 'American':
				price = option_am(self.model, self.cp,S,self.K,T,r,q,**paras)
		return price

	## plot fitted solution and benchmark
	def plot(self, S_sorted, T, r = 0.05, q=0.02, **kwargs):
		S_sorted = np.squeeze(S_sorted)[:,np.newaxis]
		x = np.log(S_sorted)
		zeros = np.zeros_like(x)
		
		default_value_dict = {}
		for key in self.model_paras:
			temp = self.data_range['others'][key]
			default_value_dict[key] = (temp['high'] + temp['low'])/2
		default_value_dict.update(kwargs)

		value_list = [default_value_dict[key] for key in self.model_paras]
		value_dict = {key: default_value_dict[key] for key in self.model_paras}

		y = self.net([x, zeros + T, zeros + r, zeros + q, zeros + np.squeeze(value_list)])
		price = self.benchmark(S_sorted, T=T, r=r, q=q, **value_dict)
		plt.figure()
		plt.plot((S_sorted), y.numpy(), S_sorted, price)
		plt.show()
		
	## predict over a dataset
	def predict(self, test_tensor=None):
		if test_tensor is None:
			test_tensor = self.test_tensor
		y = self.net([test_tensor['x'], test_tensor['t'], 
					test_tensor['r'], test_tensor['q'], test_tensor['paras']])
		return y.numpy()
		
	## calculate test samples 
	def test_price_fun(self, test_data=None):
		if test_data is None:
			test_data = self.test_data
		iter = zip(test_data['x'], test_data['t'], 
					test_data['r'], test_data['q'], test_data['paras'])
		fft_test = []
		for (x_, t_, r_, q_, paras_) in tqdm(iter):
				price = self.benchmark(np.exp(x_),t_,r_,q_,**dict(zip(self.model_paras, paras_)))
				fft_test.append(price)
		fft_test = np.array(fft_test)
		return fft_test  

	## RMSE    
	def compare(self, y, test_price):
			return np.sqrt(np.mean((y  - test_price)**2))

	## one training step
	def train_step_raw(self, batch, optimizer, training, fix, weighted):
		model = self.net

		y_grid = self.y_grid
		weight_grid = self.weight_grid
		zeros = self.zeros
		
		low_bound = self.data_range['x']['bound']['low']
		high_bound = self.data_range['x']['bound']['high']
		K = float(self.K) 

		x_tensor = batch['x']
		t_tensor = batch['t'] 
		r_tensor = batch['r'] 
		q_tensor = batch['q'] 
		paras_tensor = batch['paras'] 
		omega_tensor = batch['omega'] 
		sig2_tensor = batch['sig2'] 
		k_p_tensor = batch['k_p'] 
		k_n_tensor = batch['k_n'] 
		r_p_tensor = batch['ratio_p'] 
		r_n_tensor = batch['ratio_n'] 
		
		## compute loss 
		with tf.GradientTape() as tape:
			with tf.GradientTape() as g:
				g.watch(x_tensor)
				with tf.GradientTape() as gg:
					gg.watch(x_tensor)
					with tf.GradientTape() as ggg:
						ggg.watch(t_tensor)
						w_tensor = model([x_tensor, t_tensor, r_tensor, q_tensor, paras_tensor])
					dw_dt = ggg.gradient(w_tensor, t_tensor) 
				dw_dx = gg.gradient(w_tensor, x_tensor)     
			d2w_dx2 = g.gradient(dw_dx, x_tensor)  

			duplicated_paras = [tf.repeat(tensor, repeats=len(self.y_grid), axis=0) for tensor in 
													[t_tensor, r_tensor, q_tensor, paras_tensor]]
			rep_t_tensor, rep_r_tensor, rep_q_tensor, _ = duplicated_paras
			
			if self.model == 'Merton': ### special
				y_p_tensor = tf.reshape(x_tensor + batch['alpha'] + r_p_tensor * y_grid,(-1,1))
				y_n_tensor = tf.reshape(x_tensor + batch['alpha'] - r_n_tensor * y_grid,(-1,1))
			else:
				y_p_tensor = tf.reshape(x_tensor + r_p_tensor * y_grid,(-1,1))
				y_n_tensor = tf.reshape(x_tensor - r_n_tensor * y_grid,(-1,1))

			## extrapolation
			y_p_tensor_clip = tf.minimum(y_p_tensor, high_bound)
			y_n_tensor_clip = tf.maximum(y_n_tensor, low_bound)

			w_p = model([y_p_tensor_clip] + duplicated_paras)
			w_n = model([y_n_tensor_clip] + duplicated_paras)

			if self.exercise == 'American':     
				if self.cp == 'call':
					w_p += tf.maximum(tf.exp(y_p_tensor)-np.exp(high_bound),0)
				elif self.cp == 'put':
					w_n += tf.maximum(np.exp(low_bound) - tf.exp(y_n_tensor),0)
			elif self.exercise == 'European':        
				if self.cp == 'call':
					w_p += tf.maximum(tf.exp(y_p_tensor)-np.exp(high_bound),0) * tf.exp(-rep_q_tensor*rep_t_tensor)
				elif self.cp == 'put':
					w_n += tf.maximum(np.exp(low_bound) - tf.exp(y_n_tensor),0) * tf.exp(-rep_q_tensor*rep_t_tensor)

			w_p = tf.reshape(w_p, (-1,len(self.y_grid)))
			w_n = tf.reshape(w_n, (-1,len(self.y_grid)))
			int_p = tf.reduce_sum((w_p - w_tensor)*k_p_tensor*r_p_tensor*weight_grid, axis = 1, keepdims=True)
			int_n = tf.reduce_sum((w_n - w_tensor)*k_n_tensor*r_n_tensor*weight_grid, axis = 1, keepdims=True)
			
			## PIDE
			if fix:
				diff = (r_tensor-q_tensor-sig2_tensor/2) * dw_dx + sig2_tensor/2 * d2w_dx2 - dw_dt - r_tensor * w_tensor + tf.stop_gradient(int_p + int_n+ omega_tensor*dw_dx)
			else:
				diff = (r_tensor-q_tensor+omega_tensor-sig2_tensor/2) * dw_dx + sig2_tensor/2 * d2w_dx2 - dw_dt - r_tensor * w_tensor + int_p + int_n

			## initial condition
			if self.cp == 'call':
				x_bound = tf.maximum(tf.exp(x_tensor) - K, 0)
			elif self.cp == 'put':
				x_bound = tf.maximum(K - tf.exp(x_tensor), 0)
			x_bound_diff = model([x_tensor, zeros, r_tensor, q_tensor, paras_tensor]) - x_bound              

			## boundary condition
			if self.exercise == 'American':     
				if self.cp == 'call':
					t_bound = 0
					t_bound_2 = (-K + np.exp(high_bound))
					aux = tf.exp(x_tensor) - K - w_tensor
				elif self.cp == 'put':
					t_bound = (K - np.exp(low_bound))
					t_bound_2 = 0
					aux = K - tf.exp(x_tensor) - w_tensor
				coef = 1
				diff = tf.maximum(coef * aux, diff)

			elif self.exercise == 'European':        
				if self.cp == 'call':
					t_bound = 0
					t_bound_2 = (-K * tf.exp(-r_tensor*t_tensor) + np.exp(high_bound) * tf.exp(-q_tensor*t_tensor))
				elif self.cp == 'put':
					t_bound = (K * tf.exp(-r_tensor*t_tensor) - np.exp(low_bound) * tf.exp(-q_tensor*t_tensor))
					t_bound_2 = 0

			t_bound_diff = model([zeros + low_bound, t_tensor, r_tensor, q_tensor, paras_tensor]) - t_bound
			t_bound_diff_2 = model([zeros + high_bound, t_tensor, r_tensor, q_tensor, paras_tensor]) - t_bound_2

			## compensator for call options
			call_mul = 1.0
			call_thresh = 2.0
			if self.cp == 'call':
				t_bound_diff_2 = t_bound_diff_2/np.exp((high_bound-np.log(call_thresh*K))*call_mul)
				x_bound_diff = x_bound_diff / tf.maximum(1.0, tf.exp((x_tensor-tf.math.log(call_thresh*K))*call_mul))
				diff = diff/tf.maximum(1.0, tf.exp((x_tensor-tf.math.log(call_thresh*K))*call_mul))

			## weighted loss focuses on larger losses
			if weighted:
				loss = diff**2 + x_bound_diff**2+ t_bound_diff**2+ t_bound_diff_2**2
				loss_weight = tf.stop_gradient(loss)**float(weighted)
				loss = tf.reduce_mean(loss_weight*loss)/tf.reduce_mean(loss_weight)
				loss_root = tf.sqrt(loss)
			else:
				loss = tf.reduce_mean(diff**2 + x_bound_diff**2+ t_bound_diff**2+ t_bound_diff_2**2)
				loss_root = tf.sqrt(loss)

		if training:
			## compute gradient 
			grads = tape.gradient(loss, model.trainable_variables)
			g_bound = 10.0 if self.exercise == 'American' else 100.0
			grads, global_norm = tf.clip_by_global_norm(grads, g_bound)     
			
			## backprop
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
		return loss_root

	## training function
	def train(self, opt=tf.keras.optimizers.Adam(learning_rate=0.001), n_epochs = 30, batch_size = 200, 
						fix = False, weighted=False, plot_paras = []):  
		
		train_batches = tf.data.Dataset.from_tensor_slices(self.train_tensor).batch(batch_size,drop_remainder=True) 
		test_batches = tf.data.Dataset.from_tensor_slices(self.test_tensor).batch(batch_size,drop_remainder=True) 

		self.zeros = tf.zeros((batch_size,1), dtype = self.dtype)
		
		x_sorted = np.sort(self.test_tensor['x'],axis=0)
		S_sorted = np.exp(x_sorted)

		train_loss_list = []
		val_loss_list =[]
		train_step = tf.function(self.train_step_raw) 
		
		for epoch in range(0, n_epochs):
				start_time = time.time()
				loss_list = []
				val_list = []

				for batch in tqdm(train_batches):
						loss_root = train_step(batch, opt, training=True, fix=fix, weighted = weighted)
						loss_list.append(loss_root)
																 
				for batch in tqdm(test_batches):
						loss_root = train_step(batch, opt, training=False, fix=fix, weighted = weighted)
						val_list.append(loss_root)              
						
				print("===========================================")
				print("Epoch {:03d}: Train Loss: {:.5g}, Test Loss: {:.5g} ".format(epoch, np.mean(loss_list), np.mean(val_list)))
				print('Time for epoch {:03d} is {:.5g} sec'.format(epoch, time.time()-start_time))
				print("===========================================")   
				train_loss_list.append(float(np.mean(loss_list)))
				val_loss_list.append(float(np.mean(val_list)))
				
				## plot fitted solution during training
				for each in plot_paras:
						self.plot(S_sorted = S_sorted, **each)

		return train_loss_list, val_loss_list
