import csv
import os

script_dir = os.path.dirname(__file__)

def load_field_gradients(filename):
    field_gradients = []
    i = 0
    j = 0

    filename = os.path.join(script_dir, filename)

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if j % 2 == 0 or j == 0:
                field_gradients.append([])
                for value in row:
                    field_gradients[i].append(float(value))
                i += 1
            j += 1
    return field_gradients

def load_field_strength(filename):
    field_strengths = []
    i = 0
    j = 0

    filename = os.path.join(script_dir, filename)
    
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if j % 2 == 0 or j == 0:
                field_strengths.append([])
                for value in row:
                    field_strengths[i].append(float(value))
                i += 1
            j += 1
    return field_strengths

def get_field_values(particle, field_strengths, field_gradients, x_max, y_max):
    #use particle position to calculate index and return field gradient value
    pos = particle.position
    
    j = int(pos[0] // (x_max / len(field_gradients[0])))
    i = int(pos[1] // (y_max / len(field_gradients)))
    #print(pos)
    return field_strengths[i][j], field_gradients[i][j]
