import tensorflow as tf
import random
import numpy as np
import pandas as pd
import copy
import time
import pickle
import seaborn as sns
from keras import backend
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Network(object):
    def __init__(self, sizes):

        '''The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.'''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # helper variables
        self.bias_nitem = sum(sizes[1:])
        self.weight_nitem = sum([self.weights[i].size for i in range(self.num_layers - 2)])

    def feedforward(self, a):
        '''Return the output of the network if ``a`` is input.'''
        for i in range(len(self.weights)-1):
            a = self.sigmoid(np.dot(self.weights[i], a) + self.biases[i])
        a = np.dot(self.weights[-1], a) + self.biases[-1]
        return a

    def sigmoid(self, z):
        '''The sigmoid function.'''
        return 1.0 / (1.0 + np.exp(-z))

    def score(self, X, y):
        total_score = 0
        for i in range(X.shape[0]):
            predicted = self.feedforward(X[i].reshape(-1, 1))
            actual = y[i].reshape(-1, 1)
            total_score += np.sum(np.power(predicted - actual, 2) / 2)  # mean-squared error
        return total_score

    def r2_score(self, X, y):
        rss, tss = 0, 0.00001
        for i in range(X.shape[0]):
            output = self.feedforward(X[i].reshape(-1, 1))
            rss += np.square(output - y[i])
            tss += np.square(output - np.mean(y))
        r2_score = (1 - rss/tss)
        return r2_score

    def __str__(self):
        s = "\nBias:\n\n" + str(self.biases)
        s += "\nWeights:\n\n" + str(self.weights)
        s += "\n\n"
        return s

    def test(self, X_test, y_test):
        y_pred = []
        for i in range(X_test.shape[0]):
            output = self.feedforward(X_test[i].reshape(-1, 1))
            y_pred.append(output)
        y_true = [y for y in y_test]
        err = [(y_true[i] - y_pred[i]) for i in range(len(y_true))]
        index_n = [i for i in range(len(y_true))]
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'error(MPa)': err}, index=index_n)
        print(df)
        print("------------------------------------")
        initial_weight = self.weights
        # print(initial_weight)
        initial_biases = self.biases
        # print(initial_biases)

        y_true_test = np.array(y_true).reshape(-1,1)
        y_predict_test = np.array(y_pred).reshape(-1,1)

        regression_test = LinearRegression(fit_intercept=False).fit(y_true_test, y_predict_test)
        R2e = r2_score(y_true_test, y_predict_test)
        R = str(np.sqrt(R2e))[:6]
        plt.text(35, 70, "R = " + R, color='red')
        plt.scatter(y_true_test, y_predict_test)
        plt.plot(y_true_test, regression_test.predict(y_true_test), color='red', linewidth=3)
        plt.title('Predict vs True - Test Data - GA')
        plt.xlabel('True')
        plt.ylabel('Predict')
        plt.savefig('Save_png/1.png', format='png', dpi=300)
        plt.show()
        return initial_biases, initial_weight

class NNGeneticAlgo:

    def __init__(self, n_pops, net_size, mutation_rate, crossover_rate, retain_rate, X, y):

        '''
        n_pops   = How much population do our GA need to create
        net_size = Size of neural network for population members
        mutation_rate = probability of mutating all bias & weight inside our network
        crossover_rate = probability of cross-overing all bias & weight inside out network
        retain_rate = How many to retain our population for the best ones
        X = our data to test accuracy
        y = our data-label to test accuracy
        '''
        self.n_pops = n_pops
        self.net_size = net_size
        self.nets = [Network(self.net_size) for i in range(self.n_pops)]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.retain_rate = retain_rate
        self.X = X[:]
        self.y = y[:]

    def get_random_point(self, type):

        '''
        @type = either 'weight' or 'bias'
        @returns tuple (layer_index, point_index)
            note: if type is set to 'weight', point_index will return (row_index, col_index)
        '''
        nn = self.nets[0]
        layer_index, point_index = random.randint(0, nn.num_layers - 2), 0
        if type == 'weight':
            row = random.randint(0, nn.weights[layer_index].shape[0] - 1)
            col = random.randint(0, nn.weights[layer_index].shape[1] - 1)
            point_index = (row, col)
        elif type == 'bias':
            point_index = random.randint(0, nn.biases[layer_index].size - 1)
        return (layer_index, point_index)

    def get_all_scores(self):
        return [net.score(self.X, self.y) for net in self.nets]

    def get_all_r2_score(self):
        return [net.r2_score(self.X, self.y) for net in self.nets]

    def crossover(self, father, mother):

        '''
        @father = neural-net object representing father
        @mother = neural-net object representing mother
        @returns = new child based on father/mother genetic information
        '''
        # make a copy of father 'genetic' weights & biases information
        nn = copy.deepcopy(father)

        # cross-over bias
        for _ in range(self.nets[0].bias_nitem):
            # get some random points
            layer, point = self.get_random_point('bias')
            # replace genetic (bias) with mother's value
            if random.uniform(0, 1) < self.crossover_rate:
                nn.biases[layer][point] = mother.biases[layer][point]

        # cross-over weight
        for _ in range(self.nets[0].weight_nitem):
            # get some random points
            layer, point = self.get_random_point('weight')
            # replace genetic (weight) with mother's value
            if random.uniform(0, 1) < self.crossover_rate:
                nn.weights[layer][point] = mother.weights[layer][point]

        return nn

    def mutation(self, child):

        '''
        @child_index = neural-net object to mutate its internal weights & biases value
        @returns = new mutated neural-net
        '''
        nn = copy.deepcopy(child)
        # mutate bias
        for _ in range(self.nets[0].bias_nitem):
            # get some random points
            layer, point = self.get_random_point('bias')
            # add some random value between -0.5 and 0.5
            if random.uniform(0, 1) < self.mutation_rate:
                nn.biases[layer][point] += random.uniform(-0.5, 0.5)
        # mutate weight
        for _ in range(self.nets[0].weight_nitem):
            # get some random points
            layer, point = self.get_random_point('weight')
            # add some random value between -0.5 and 0.5
            if random.uniform(0, 1) < self.mutation_rate:
                nn.weights[layer][point[0], point[1]] += random.uniform(-0.5, 0.5)
        return nn

    def evolve(self):
        # calculate score for each population of neural-net
        score_list = list(zip(self.nets, self.get_all_scores()))

        # sort the network using its score
        score_list.sort(key=lambda x: x[1])

        # exclude score as it is not needed anymore
        score_list = [obj[0] for obj in score_list]

        # keep only the best one
        retain_num = int(self.n_pops * self.retain_rate)
        score_list_top = score_list[:retain_num]

        # return some non-best ones
        retain_non_best = int((self.n_pops - retain_num) * self.retain_rate)
        for _ in range(random.randint(0, retain_non_best)):
            score_list_top.append(random.choice(score_list[retain_num:]))

        # breed new childs if current population number less than what we want
        while len(score_list_top) < self.n_pops:
            father = random.choice(score_list_top)
            mother = random.choice(score_list_top)

            if father != mother:
                new_child = self.crossover(father, mother)
                new_child = self.mutation(new_child)
                score_list_top.append(new_child)

        # copy our new population to current object
        self.nets = score_list_top
    def print_test(self, X_test, y_test):
        initial_biases, initial_weight = self.nets[-1].test(X_test, y_test)
        return initial_biases, initial_weight
class my_network():
    def __init__(self, initial_biases,initial_weight,X, y, X_test, y_test, X_val, y_val, X_train, y_train):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.biases = initial_biases
        self.weights = initial_weight

    def rmse(self, y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    def mae(self, y_true, y_pred):
        return backend.mean(backend.abs(y_pred - y_true), axis=-1)

    def ANN(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32, input_dim=6, use_bias=True, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(16, use_bias=True, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(1))
        model.summary()
        opt = tf.keras.optimizers.RMSprop(learning_rate = 0.015, rho = 0.9)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=[self.mae, self.rmse])
        return model

    def train_ANN(self):
        model = self.ANN()
        for i in range(len(self.weights)):
            w = np.array(self.weights[i]).transpose()
            temp = []
            for j in self.biases[i]:
                temp.append(j[0])
            vector = np.vectorize(np.float)
            b = vector(temp)
            t = np.array(model.layers[i].get_weights()[0])
            d = np.array(model.layers[i].get_weights()[1])
            model.layers[i].set_weights([w,b])

        # Tạo callback
        # callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        filepath = "Save_Model/checkpoint_ANN_GA.hdf5"
        callback = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='auto')
        model.fit(self.X_train, self.y_train, epochs=1000, validation_split=0.1, validation_data=(self.X_val, self.y_val), callbacks=[callback])
        model.evaluate(self.X_test, self.y_test)

        #y_predict_test = [y[0] for y in model.predict(self.X_test).tolist()]
        y_predict_test = model.predict(self.X_test)
        y_predict_train = [y for y in model.predict(self.X_train)]
        y_true_test = np.array(self.y_test).reshape(-1,1)
        y_predict_total = model.predict(self.X)
        y_true_total = np.array(self.y).reshape(-1,1)


        y_pred = [y for y in y_predict_test]
        y_true = [y for y in self.y_test]
        err = [(y_true[i] - y_pred[i]) for i in range(len(y_true))]
        index_n = [i for i in range(len(y_true))]
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'error(MPa)': err}, index=index_n)
        print(df)

        regression_test = LinearRegression(fit_intercept=False).fit(y_true_test, y_predict_test)
        R2e = r2_score(y_true_test, y_predict_test)
        R = str(np.sqrt(R2e))[:6]
        plt.text(35, 70, "R = " + R, color='red')
        plt.scatter(self.y_test, y_predict_test)
        plt.plot(y_true_test, regression_test.predict(y_true_test), color='red', linewidth=3)
        plt.title('Predict vs True - TEST DATA - Final')
        plt.xlabel('True')
        plt.ylabel('Predict')
        plt.savefig('Save_png/2.png', format='png', dpi=300)
        plt.show()

        regression_total = LinearRegression(fit_intercept=False).fit(y_true_total, y_predict_total)
        R2e = r2_score(y_true_total, y_predict_total)
        R = str(np.sqrt(R2e))[:6]
        plt.text(35, 75, "R = " + R, color='red')
        plt.scatter(self.y, y_predict_total)
        plt.plot(y_true_total, regression_total.predict(y_true_total), color='red', linewidth=3)
        plt.title('Predict vs True - ALL DATA - Final')
        plt.xlabel('True')
        plt.ylabel('Predict')
        plt.savefig('Save_png/3.png', format='png', dpi=300)
        plt.show()

        y_pred_t = [y for y in y_predict_total]
        y_true_t = [y for y in self.y]
        err_t = [(y_true_t[i] - y_pred_t[i]) for i in range(len(y_true_t))]
        index_n_t = [i for i in range(len(y_true_t))]
        df_t = pd.DataFrame({'y_true': y_true_t, 'y_pred': y_pred_t, 'error(MPa)': err_t}, index=index_n_t)

        plt.plot(df_t['y_true'], linestyle='--', color='red')
        plt.plot(df_t['y_pred'])
        plt.ylabel('Cuong do (MPa)')
        plt.xlabel('So luong mau test')
        plt.title('So sanh R_thucte va R_dudoan - Total')
        plt.savefig('Save_png/4.png', format='png', dpi=300)
        plt.show()

        y_pred = np.array(model.predict(self.X)).reshape(-1, 1)
        y_true = np.array(self.y).reshape(-1, 1)
        err = (y_true - y_pred)

        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6), dpi=120)
        sns.distplot(err, hist=True, kde=True, color='darkblue', hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 2}, ax=ax1)
        ax2 = ax1.twinx()
        sns.histplot(err, ax=ax2)
        ax1.set_xlabel('Error (MPa)', fontsize=10)
        ax2.set_ylabel('Số lượng', color='red', fontsize=10)
        ax1.set_ylabel('Tần suất', color='blue', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='red')
        plt.savefig('Save_png/5.png', format='png', dpi=120)
        plt.show()

        R2e = r2_score(self.y_train, y_predict_train)
        print("\nPERFORMANCE IN TRAINING")
        # print("Coefficient of Determination: ", R2e)
        print("Correlation Coefficient: ", np.sqrt(R2e))
        print("MSE: ", mean_squared_error(self.y_train, y_predict_train))
        print("MAE: ", mean_absolute_error(self.y_train, y_predict_train))

        R2 = r2_score(self.y_test, y_predict_test)
        print("---------------------------------------------")
        print("\nPERFORMANCE IN TEST")
        # print("Coefficient of Determination: ", R2)
        print("Correlation Coefficient: ", np.sqrt(R2))
        print("MSE: ", mean_squared_error(self.y_test, y_predict_test))
        print("MAE: ", mean_absolute_error(self.y_test, y_predict_test))

def main():
    # load data
    df = pd.read_csv('Data/BT_CKDKHH_Cuongdo_1.csv')
    # df.info()
    # print(df.describe())
    # print(df.isnull().sum())

    X = df.drop('Cuongdo', axis=1)
    y = df['Cuongdo']

    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    # sc = MinMaxScaler()
    sc = StandardScaler()
    X = sc.fit_transform(X.values)
    pickle.dump(sc, open('Save_Model/scaler_ANN_GA.pkl', 'wb'))

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2 , random_state=10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=20)

    # parameters
    N_POPS = 30
    NET_SIZE = [6, 32, 16, 1]
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.4
    RETAIN_RATE = 0.4

    # start our neural-net & optimize it using genetic algorithm
    nnga = NNGeneticAlgo(N_POPS, NET_SIZE, MUTATION_RATE, CROSSOVER_RATE, RETAIN_RATE, np.array(X_test), np.array(y_test))

    start_time = time.time()

    # run for n iterations
    for i in range(500):

        if i % 20 == 0:
            print("Current iteration : {}".format(i + 1))
            print("Time taken by far : %.1f seconds" % (time.time() - start_time))
            print("Current top member's network R2_score: %.2f\n" % nnga.get_all_r2_score()[0])

        # evolve the population
        nnga.evolve()
    initial_biases, initial_weight = nnga.print_test(X_test, y_test)
    print("====================================")
    ann = my_network(initial_biases, initial_weight, X, y, X_test, y_test, X_val, y_val, X_train, y_train)
    ann.train_ANN()

if __name__ == "__main__":
    main()