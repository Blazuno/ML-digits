import pygame
import numpy as np
import random
import os
import time
import sys


#activation function class
class activation_function:
    def __init__(self, func, deriv_func):
        self.func = func
        self.deriv_func = deriv_func


#neuron
class neuron:
    def __init__(self):
        self.incoming = []
        self.outgoing = []
        self.value = 0
        self.z_value = 0
        #backpropagation stuff
        self.node_value = 0
        self.cost = None
        self.bias = False
        #pygame drawing stuff
        self.x = None
        self.y = None
        

#connection with weights n shit
class connection:
    def __init__(self, weight):
        self.source = None
        self.target = None
        self.weight = weight
        #backpropagation stuff
        self.gradient = 0
        


#bias neuron
class bias_neuron:
    def __init__(self):
        self.value = 1
        self.outgoing = []
        self.x = None
        self.y = None
        self.bias = True

#data point for backpropagation
class data_point:
    def __init__(self, inputs, expected_outputs):
        self.inputs = inputs
        self.expected_outputs = expected_outputs

#te layer
class layer:
    def __init__(self, size):
        self.neurons = []
        for i in range(size):
            self.neurons.append(neuron())



#the whole network
class network:  
    #sigmoid function
    
    def sig(x):
        return 1/(1 + np.exp(-1 * x))
    #derivative of sigmoid
    def deriv_sig(x):
        sig_x = network.sig(x)
        return sig_x * (1 - sig_x)
    sig_func = activation_function(sig, deriv_sig)
    def relu(x):
        if x <= 0:
            return 0
        else:
            return x
    def deriv_relu(x):
        if x <= 0:
            return 0
        else:
            return 1
    relu_func = activation_function(relu, deriv_relu)
    #cost function
    def cost(x,y):
        return pow(x - y, 2)
    #derivative of cost function
    def deriv_cost(x,y):
        return 2 * (x-y)
    #creation of network
    def __init__(self, structure, bias, activation, output_activation):
        self.layers = []
        self.activation = activation
        self.output_activation = output_activation
        if bias:
            self.bias_neuron = bias_neuron()
        for i in structure:
            self.layers.append(layer(i))
        for i in range(1, len(self.layers)):
            for old_neuron in self.layers[i-1].neurons:
                for new_neuron in self.layers[i].neurons:
                    random_weight = round(random.uniform(-1,1), 2)
                    new_connection = connection(random_weight)
                    old_neuron.outgoing.append(new_connection)
                    new_neuron.incoming.append(new_connection)
                    new_connection.source = old_neuron
                    new_connection.target = new_neuron
        #bias neuron implementation
        if bias:
            for i in range(1, len(self.layers)):
                for neuron in self.layers[i].neurons:
                    random_weight = round(random.uniform(-1,1), 2)
                    new_connection = connection(random_weight)
                    neuron.incoming.append(new_connection)
                    self.bias_neuron.outgoing.append(new_connection)
                    new_connection.source = self.bias_neuron
                    new_connection.target = neuron

    def compute_inputs(self, inputs):
        #resetting the inputs for the computation
        for neuron in self.layers[0].neurons:
            neuron.value = 0

        iterator = 0
        for input in inputs:
            if iterator < len(self.layers[0].neurons):
                self.layers[0].neurons[iterator].value = input
                self.layers[0].neurons[iterator].z_value = input
                iterator += 1

        for i in range(1, len(self.layers)):
            
            for neuron in self.layers[i].neurons:
                total = 0
                for connection in neuron.incoming:
                    total += connection.weight * connection.source.value
                neuron.z_value = total
                if i == len(self.layers) - 1:
                    total = self.output_activation.func(total)
                else:
                    total = self.activation.func(total)
                neuron.value = total          
            
        
        outputs = []
        for neuron in self.layers[len(self.layers)-1].neurons:
            outputs.append(round(neuron.value,2))
        return outputs
    #NEEDS PYGAME
    def draw_network(self, surface, start_pos, neuron_size, connection_size, xy_factors):
        new_pos_x = start_pos[0]
        new_pos_y = start_pos[1]
        x_factor = xy_factors[0]
        y_factor = xy_factors[1]
        #drawing neurons
        first_iter = True 
        for layer in self.layers:
            if first_iter:
                for neuron in layer.neurons:
                    neuron.x = new_pos_x
                    neuron.y = new_pos_y
                    pygame.draw.circle(surface, (255,255,255), (new_pos_x, new_pos_y), neuron_size)
                    new_pos_y += neuron_size * y_factor
                    self.bias_neuron.x = new_pos_x
                    self.bias_neuron.y = new_pos_y
                    pygame.draw.circle(surface, (255,255,255), (new_pos_x, new_pos_y), neuron_size)
                    first_iter = False
            else:
                first_y = self.layers[0].neurons[0].y
                second_y = self.bias_neuron.y
                center_y = (first_y + second_y)/2
                first_y_length = neuron_size * y_factor * len(self.layers[0].neurons)
                second_y_length = neuron_size * y_factor * (len(layer.neurons) - 1)
                y_increment = (second_y_length - first_y_length) / 2
                new_pos_y = self.layers[0].neurons[0].y - y_increment
                for neuron in layer.neurons:
                    neuron.x = new_pos_x
                    neuron.y = new_pos_y
                    pygame.draw.circle(surface, (255,255,255), (new_pos_x, new_pos_y), neuron_size)
                    new_pos_y += neuron_size * y_factor
            new_pos_x += neuron_size * x_factor
            new_pos_y = start_pos[0]
        #drawing connections
        for i in range(1, len(self.layers)):
            for neuron in self.layers[i].neurons:
                for con in neuron.incoming:
                    pygame.draw.line(surface, (255,0,255), (con.source.x, con.source.y), (con.target.x, con.target.y), connection_size)
    #gets the overall cost of the network with respect to the training data
    def get_network_cost(self, training_data):
        total_cost = 0
        iter = 1
        for data_point in training_data:
            outputs = self.compute_inputs(data_point.inputs)
            iter += 1
            for i in range(len(outputs)):
                total_cost += network.cost(outputs[i], data_point.expected_outputs[i])
        return total_cost / len(training_data)
    #returns all weights within the network
    def get_outputs(self):
        outputs = []
        for neuron in self.layers[len(self.layers) - 1].neurons:
            outputs.append(neuron.value)
        return outputs
    def get_all_weights(self):
        weights = []
        for i in range(1, len(self.layers)):
            for neuron in self.layers[i].neurons:
                for connection in neuron.incoming:
                    weights.append(connection.weight)
        return weights
    #returns all the gradients of each connection of the network
    def get_all_gradients(self):
        gradients = []
        for layer in self.layers:
            for neuron in layer.neurons:
                for connection in neuron.incoming:
                    gradients.append(round(connection.gradient, 2))
        return gradients
    #returns all the node values of each neuron of the network
    def get_all_node_values(self):
        node_values = []
        for i in range(1, len(self.layers)):
            for neuron in self.layers[i].neurons:
                node_values.append(neuron.node_value)
        return node_values
    #updating the gradients of each connection using backpropagation through derivatives
    def backpropagation(self, training_data):
        #resetting each gradient
        for layer in self.layers:
                for neuron in layer.neurons:
                    for connection in neuron.incoming:
                        connection.gradient = 0
        for data_point in training_data:
            first_iter = True
            #first we use the data point to set the output values of the network
            outputs = self.compute_inputs(data_point.inputs)
            #now we backpropagate to set the gradients
            for i in range(len(self.layers) - 1, 0, -1):
                #setting node values for output layer
                if first_iter:
                    for i2 in range(len(self.layers[i].neurons)):
                        #node value = derivative of cost * derivative of activation
                        self.layers[i].neurons[i2].node_value = network.deriv_cost(self.layers[i].neurons[i2].value, data_point.expected_outputs[i2]) * self.output_activation.deriv_func(self.layers[i].neurons[i2].z_value)
                    #setting gradients for output layer
                    for neuron in self.layers[i].neurons:
                        for connection in neuron.incoming:
                            #gradient is the node value * the activation of the source of the connection
                            connection.gradient += neuron.node_value * connection.source.value 
                    first_iter = False
                #setting node values and gradients for rest of layers
                elif i != len(self.layers) - 1:
                    #setting node values
                    for neuron in self.layers[i].neurons:
                        #this time we need to add all the node values of the previous layer multiplied by the weight connecting the 2 nodes
                        total = 0
                        for connection in neuron.outgoing:
                            total += connection.weight * connection.target.node_value
                        total *= self.activation.deriv_func(neuron.z_value)
                        neuron.node_value = total
                    #setting gradients
                    for neuron in self.layers[i].neurons:
                        for connection in neuron.incoming:
                                #same concept as before
                                connection.gradient += neuron.node_value * connection.source.value
    #applies all gradients
    def apply_all_gradients(self, learn_rate, training_data):
        for layer in self.layers:
            for neuron in layer.neurons:
                for connection in neuron.incoming:
                    #learn rate averaged by the training data
                    connection.weight -= connection.gradient * (learn_rate / len(training_data))
    #trains the network using training data + learn rate
    def train_via_gradient_descent(self, training_data, learn_rate, screen = False, font = False, pixel_values = False):
        while True:
            #sets gradients
            self.backpropagation(training_data)
            #applies gradients
            self.apply_all_gradients(learn_rate, training_data)
            gradients = self.get_all_gradients()
            #finishes learning when all the rounded gradients are equivalent to 0
            if gradients.count(0) == len(gradients):
                break
            os.system("cls")
            if screen:
                fill_part_of_screen((560, 0, 400, 560))
                fill_part_of_screen((0, 560, 560, 300))
                draw_cost()
                draw_all_outputs(self.get_outputs())
                pygame.display.flip()
                



            



                   




#making network
my_network = network([784, 50, 10], True, network.sig_func, network.sig_func)




# Initialize Pygame
pygame.init()

# Set up display
screen = pygame.display.set_mode((28*20 + 400, 28*20 + 300))
pygame.display.set_caption("Pixel Drawer")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create a 32x32 grid
pixel_size = 20

# Initialize a 1D list to store pixel values (0 or 1)
pixel_values = [0] * (28 * 28)

#training data can be whatever you want as long as its consistent with # of inputs and outputs in the network
training_data = [
    data_point(pixel_values, [0] * 10)
]


#create ze font
font = pygame.font.Font(None, 36)
def draw_grid():
    for x in range(0, 28*pixel_size, pixel_size):
        pygame.draw.line(screen, WHITE, (x, 0), (x, 28*pixel_size))
        pygame.draw.line(screen, WHITE, (0, x), (28*pixel_size, x))
    pygame.draw.line(screen, WHITE, (0, 0), (28*pixel_size, 0))  # Top border
    pygame.draw.line(screen, WHITE, (0, 28*pixel_size-1), (28*pixel_size, 28*pixel_size-1))  # Bottom border
    pygame.draw.line(screen, WHITE, (0, 0), (0, 28*pixel_size))  # Left border
    pygame.draw.line(screen, WHITE, (28*pixel_size-1, 0), (28*pixel_size-1, 28*pixel_size))  # Right border

def draw_all_outputs(outputs):
    i = 0
    x_pos = 600
    y_pos = 50
    for output in outputs:
        text = str(i) + " confidence: " + str(round(output * 100)) + "%"
        text_surf = font.render(text, True, (255,255,255))
        screen.blit(text_surf, (x_pos, y_pos))
        y_pos += 50
        i += 1
#making all the button rects
data_buttons = []
x_pos = 900
y_pos = 50
for i in range(0,10):
    data_buttons.append(pygame.Rect(x_pos, y_pos, 25, 25))
    y_pos += 50
def draw_all_data_buttons():
    for rect in data_buttons:
        pygame.draw.rect(screen, (0,255,0), rect)


def fill_part_of_screen(rect):
    pygame.draw.rect(screen, (0,0,0), rect)

#draw train button
def draw_train_button():
    pygame.draw.rect(screen, (0,255,0), pygame.Rect(850, 600, 100, 25))
    text_surf = font.render("Train", True, (255,255,255))
    screen.blit(text_surf, (852, 602))

#draw cost value
def draw_cost():
    text = "Cost: " + str(my_network.get_network_cost(training_data))
    text_surf = font.render(text, True, (255,255,255))
    screen.blit(text_surf, (50, 600))

#draw successfully added
def draw_added(i):
    text = "Successfully added " + str(i)
    text_surf = font.render(text, True, (255,255,255))
    screen.blit(text_surf, (50, 650))

def clear_canvas():
    screen.fill(BLACK)
    for i in range(28 * 28):
        pixel_values[i] = 0

def save_to_list(x, y):
    i, j = y // pixel_size, x // pixel_size
    index = i * 28 + j
    if index < len(pixel_values):
        pixel_values[index] = 1

def main():
    drawing = False

    
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                pos = pygame.mouse.get_pos()
                for i in range(len(data_buttons)):
                    data_button = data_buttons[i]
                    if data_button.x < pos[0] < data_button.x + data_button.width and data_button.y < pos[1] < data_button.y + data_button.height:
                        expected_outputs = [0] * 10
                        expected_outputs[i] = 1
                        training_data.append(data_point(pixel_values, expected_outputs))
                if 850 < pos[0] < 850 + 100 and 600 < pos[1] < 600 + 25:
                    my_network.train_via_gradient_descent(training_data, 0.1, screen, font, pixel_values) 
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False

            if drawing:
                pos = pygame.mouse.get_pos()
                x, y = pos[0] // pixel_size * pixel_size, pos[1] // pixel_size * pixel_size
                if x <= 540 and y <= 540:
                    pygame.draw.rect(screen, WHITE, (x, y, pixel_size, pixel_size))
                    save_to_list(x, y)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    clear_canvas()

        draw_grid()
        fill_part_of_screen((560, 0, 400, 560))
        fill_part_of_screen((0, 560, 560, 300))
        outputs = my_network.compute_inputs(pixel_values)
        draw_all_outputs(outputs)
        draw_all_data_buttons()
        draw_train_button()
        draw_cost()
        pygame.display.flip()
main()

