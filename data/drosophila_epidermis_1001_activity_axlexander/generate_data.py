# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:52:30 2024

@author: hcwan
"""




import pandas as pd

def read_fa(file_name):
    sequence = []
    with open(file_name) as f:
        for item in f:
            if '>' not in item:
                sequence.append(item.strip('\n'))
    return sequence


def generate_activity(end):
    file_activity = 'fold09_sequences_activity_' + end + '.txt'
    file_sequence = 'fold09_sequences_' + end + '.fa'
    
    sequence = read_fa(file_sequence)
    activity = pd.read_table(file_activity, sep= '	')
    
    
    with open(end + '_seq.txt','w') as f:
        for item in sequence:
            f.write(item + '\n')
            
    all_name = list(activity)
    
    for name in all_name:
        if name == 'epidermis':   
          with open(end + '_exp_' + name + '.txt', 'w') as f:
              for item in activity[name]:
                  f.write(str(item) + '\n')


if __name__ == '__main__':   
    end = 'Train'
    generate_activity(end)
    end = 'Val'
    generate_activity(end)
    end = 'Test'
    generate_activity(end)
