from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json
import pygame
from car import Car
from math_util import *
from action_handler import *

class NeuroEvolutionary:
    def __init__(self,  
                 shape=31,  # kinematic_ga
                 num_actions=8):
        self.kinematic_ga = KinematicGA(shape, num_actions)
        self.car = Car(5, 27)
        self.predicted_action = -1
        self.clock = pygame.time.Clock()
        self.ticks = 60
        

    def run_ga(self, nn_model, dists):
        predicted_action = -1
        dt = self.clock.get_time()/1000
        if predicted_action != Actions.reverse.value:
            apply_action(predicted_action, self.car, dt)
        print(dt)
        self.car.update(dt)
        sensor_distances = np.asarray(dists.data)
        input_data = np.append(sensor_distances, self.car.velocity[0])
        input_data_tensor = np.reshape(input_data, (1, input_data.shape[0]))
        print(input_data_tensor)
        prediction = nn_model.predict(input_data_tensor)
        #prediction = nn_model._make_predict_function(input_data_tensor)
        predicted_action = np.argmax(prediction[0])
        self.clock.tick(self.ticks)
        return self.car.steering, self.car.velocity[0]
    
class KinematicGA(object):
    def __init__(self, shape, num_actions):
        self.shape = shape
        self.model = self.build_classifier(shape + 1, num_actions)
        self.valid_layer_names = ['hidden1', 'hidden2', 'hidden3']
        self.layer_weights, self.layer_shapes = self.init_shapes()

    def build_classifier(self, shape, num_actions):
        # create classifier to train
        classifier = Sequential()

        classifier.add(
            Dense(units=6, input_dim=shape, activation='relu', name='hidden1', kernel_initializer='glorot_uniform',
                  bias_initializer='zeros'))

        classifier.add(Dense(units=7, activation='relu', kernel_initializer='glorot_uniform', name='hidden2',
                             bias_initializer='zeros'))

        classifier.add(
            Dense(units=int(num_actions), activation='softmax', kernel_initializer='glorot_uniform', name='hidden3',
                  bias_initializer='zeros'))

        # Compile the CNN
        classifier.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return classifier

    def init_shapes(self):
        layer_weights = []
        layer_shapes = []
        # get layer weights
        for layer_name in self.valid_layer_names:
            layer_weights.append(self.model.get_layer(layer_name).get_weights())

        # break up the weights and biases
        # layer_weights = np.concatenate(layer_weights) ???
        layer_wb = []
        for w in layer_weights:
            layer_wb.append(w[0])
            layer_wb.append(w[1])

        # set starting index and shape of weight/bias
        for layer in layer_wb:
            layer_shapes.append(
                [0 if layer_shapes.__len__() == 0 else layer_shapes[-1][0] + np.prod(
                    layer_shapes[-1][1]), layer.shape])

        layer_weights = np.asarray(layer_wb)
        # flatten all the vectors
        layer_weights = [layer_weight.flatten() for layer_weight in layer_weights]

        # make one vector of all weights and biases
        layer_weights = np.concatenate(layer_weights)
        return layer_weights, layer_shapes
    
    def load_model(self, model_name):
        json_file = open('./used_models/ga/' + model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("./used_models/ga/" + model_name + ".h5")
        print("Loaded model from disk")
