import random
import numpy as np


class PSOWI:
    def __init__(self,
                 model,
                 dataset,
                 metirc_idx=0,
                 metirc_type='max',
                 epochs=1,
                 pso_max_iter=1000,
                 pso_non_better_iter=10,
                 pop_size=100, 
                 pop_low=0, 
                 pop_up=1,
                 layer_idxes=None):

        # 初始化
        self.model = model   # 待优化模型
        self.data_X, self.data_Y = dataset  # 可测试数据集
        self.metircs_idx = metirc_idx  # 评估指标索引
        self.metirc_type = metirc_type  # 评估指标类型
        self.epochs = epochs
        self.pso_max_iter = pso_max_iter    # 迭代的代数
        self.pso_non_better_iter = pso_non_better_iter  # 如果连续pos_non_better_iter次迭代没有提升，则停止迭代
        self.pop_size = pop_size  # 种群大小
        self.pop_low = pop_low  # 种群范围下限
        self.pop_up = pop_up  # 种群范围上限
        if layer_idxes is None:
            self.layer_idxes = [idx for idx in range(len(self.model.layers)) if self.model.layers[idx].get_weights()]
        else:
            self.layer_idxes = layer_idxes

        self.fit()
        

    def pso_one_layer(self, layer_idx):
        weights_num = len(self.model.layers[layer_idx].get_weights())
        weights_shape = []
        weights_init_tmp = []
        for i in range(weights_num):
            weights_shape.append(self.model.layers[layer_idx].get_weights()[i].shape)
            weights_init_tmp.append(self.model.layers[layer_idx].get_weights()[i].copy().reshape((-1)))

        weights_init = weights_init_tmp[0]
        for i in range(1, weights_num):
            weights_init = np.concatenate((weights_init, weights_init_tmp[i]))

        fitness_func = self.fitness_func(weights_shape, layer_idx)
        
        var_num =  sum([np.prod(shape) for shape in weights_shape])    # 变量个数

        pop_x = np.zeros((self.pop_size, var_num))    # 所有粒子的位置
        pop_v = np.zeros((self.pop_size, var_num))    # 所有粒子的速度
        p_best = np.zeros((self.pop_size, var_num))   # 每个粒子最优的位置
        p_best_fit = np.zeros((self.pop_size, 1))  
        g_best = weights_init.copy()   # 全局最优的位置
        g_best_fit = fitness_func(g_best)   # 全局最优的适应度

        # 初始化第0代初始全局最优解
        for i in range(self.pop_size):
            for j in range(var_num):
                pop_x[i][j] = weights_init[j] + np.random.normal(loc=0.0, scale=0.01, size=None)
                pop_v[i][j] = random.uniform(0, 0.1)
            p_best[i] = pop_x[i]
            p_best_fit[i] = fitness_func(p_best[i]) # 计算粒子的适应度

        # 开始迭代
        pos_non_better_iter = 0
        for iter in range(self.pso_max_iter):
            print('############ Layer {} Weights Generation {} ############'.format(layer_idx, str(iter + 1)))
            local_best = g_best.copy()
            local_best_fit = g_best_fit
            for i in range(self.pop_size):
                for j in range(var_num):
                    pop_v[i][j] = self.w * pop_v[i][j] + self.c1 * random.uniform(0, 1) * (p_best[i][j] - pop_x[i][j]) + self.c2 * random.uniform(0, 1) * (g_best[j] - pop_x[i][j])
                    pop_x[i][j] = pop_x[i][j] + pop_v[i][j]
                    if pop_x[i][j] > self.pop_up:
                        pop_x[i][j] = self.pop_up
                    elif pop_x[i][j] < self.pop_low:
                        pop_x[i][j] = self.pop_low
                tmp_fit = fitness_func(pop_x[i])
                if self.determine_better_fit(tmp_fit, local_best_fit):
                    p_best[i] = pop_x[i]
                    p_best_fit[i] = tmp_fit

                if self.determine_better_fit(tmp_fit, g_best_fit):
                    local_best = pop_x[i]
                    local_best_fit = tmp_fit

            if self.determine_better_fit(local_best_fit, g_best_fit):
                g_best = local_best
                g_best_fit = local_best_fit
                pos_non_better_iter = 0
                print('############ Layer {} Weights Update !!! ############'.format(layer_idx))
                print("############ Global Fitness: {} ############".format(g_best_fit))

            else:
                print('############ Layer {} Weights don\'t Update !!! ############'.format(layer_idx))
                print("############ Global Fitness: {} ############".format(g_best_fit))
                pos_non_better_iter += 1
                if pos_non_better_iter > self.pso_non_better_iter:
                    break

        start_idx = 0
        updated_layer_weights = []
        for i in range(1, weights_num):
            updated_weight = g_best[start_idx:start_idx + np.prod(weights_shape[i])]
            updated_weight = updated_weight.reshape(weights_shape[i])
            updated_layer_weights.append(updated_weight)
            start_idx += np.prod(weights_shape[i])

        self.model.layers[self.layer_idx].set_weights(updated_layer_weights)  # 更新权重         

    def determine_better_fit(self, fit1, fit2):
        if self.metirc_type == 'max':
            return fit1 > fit2
        else:
            return fit1 < fit2


    def fitness_func(self, weights_shape, layer_idx):
        weights_num = len(weights_shape)

        def fitness_compute(ind_var):

            start_idx = 0
            updated_layer_weights = []
            for i in range(1, weights_num):
                updated_weight = ind_var[start_idx:start_idx + np.prod(weights_shape[i])]
                updated_weight = updated_weight.reshape(weights_shape[i])
                updated_layer_weights.append(updated_weight)
                start_idx += np.prod(weights_shape[i])

            self.model.layers[layer_idx].set_weights(updated_layer_weights)  # 更新权重


            return self.model.evaluate(self.x_test, self.y_test, verbose=0)[self.metircs_idx]
        return fitness_compute

    def fit(self):
        for epoch in range(self.epochs):
            print('######################## PSO Epoch {} !!! ########################'.format(epoch))
            for layer_idx in self.layer_idxes:
                self.pso_one_layer(layer_idx)

    def get_optimal_model(self):
        return self.model