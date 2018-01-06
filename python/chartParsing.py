"""
Artifical test for chart parsing
"""

from random import shuffle
import numpy as np
import string

def generate_string(length):
    letters = list(string.ascii_lowercase) + list(string.ascii_uppercase)
    shuffle(letters)
    output = []
    for i in range(length):
        output.append(letters[i])
    return output

sen_length = 25
sentence  = generate_string(sen_length)

# Compose :  [A, B] = (A) + (B) = (AB) 
# Combine : ((AB)C), (A(BC)) = (ABC)
# A + B = (AB)
# (AB) + C = ((AB)C)

def compose(l, r):
    return "(" + l + r + ")"

def combine(list_versions):
    return list_versions[0]
    #return list_versions[0].replace("(","").replace(")","")

def compute_compositions(sent):

    length = len(sent) -1
    l_hiddens = sent[:-1]
    l_cells = sent[:-1]
    r_hiddens = sent[1:]
    r_cells = sent[1:]
    chart = []
    masks = []
    choices = []

    """
    layer_0 = []
    for i in range(len(sent)):
        layer_0.append((sent[i], sent[i]))
    chart = [layer_0]
    """
    
    chart = [sent] # list or tuple. w/e
    masks = [np.zeros(len(sent))]
    choices = [sent]
    for row in range(1, len(sent)): 
        chart.append([])
        masks.append([])
        choices.append([])
        for col in range(len(sent) - row):
            chart[row].append(None)
            masks[row].append(None)
            choices[row].append(None)
    
    for row in range(1, len(sent)): # = len(l_hiddens)
        for col in range(len(sent) - row):
            versions = []
            for i in range(row):
                #print row, col, chart[row-i-1][col], chart[i][row+col-i]
                versions.append(compose(chart[row-i-1][col], chart[i][row+col-i]))

            chart[row][col] = combine(versions)
            choices[row][col] = versions
            l = len(versions)
            rand_pos = np.random.randint(l)
            mask = np.zeros(l)
            mask[rand_pos] += 1
            masks[row][col] = mask
        
    return chart, masks, choices

chart, mask, choices = compute_compositions(sentence)

"""
for row in len(choices):
    for col in len(choices[row]):
        pick = choices[row][col][int(np.where(mask[row][col])[0])]
"""
print choices[-1][-1][int(np.where(mask[-1][-1])[0])]

