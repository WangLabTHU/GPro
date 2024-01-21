# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:52:30 2024

@author: hcwan
"""




import pandas as pd
import numpy as np

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
        with open(end + '_exp_' + name + '.txt', 'w') as f:
            for item in activity[name]:
                f.write(str(item) + '\n')

def prepare_input(fold, set, output, remove_no_ov=True, balanced_negative=True):
    # Convert sequences to one-hot encoding matrix
    file_seq = fold + "_sequences_" + set + ".fa"
    input_fasta_data_A = np.array(read_fa(file_seq))

    # output
    Activity = pd.read_table(fold + "_sequences_activity_" + set + ".txt")
    
    # Filter for active fragment tiles that overlap peaks - cleaner positive set
    if remove_no_ov:
        IDs = ((Activity[output] == "Active") & (Activity[str(output + "_overlap")] == False))
        Activity = Activity[~IDs]

        input_fasta_data_A = input_fasta_data_A[~IDs]
    
    # select a unique tile per region (or a few), in case I want to remove multiple tiles per negative region to balance the datasets
    if balanced_negative:
        # Set the Main_tile column to "No" for all rows
        Activity["Main_tile"] = "No"

        # Group the data by the ID column and apply a function to each group
        def select_random_rows(group):
            # If the group has fewer than 5 rows, select all rows
            n = 5
            if len(group) < n:
                tmp = group
            else:
                # Otherwise, select 5 random rows from the group
                tmp = group.sample(n=n)
            # Set the Main_tile column to "Yes" for the selected rows
            Activity.loc[tmp.index, "Main_tile"] = "Yes"

        Activity.groupby("ID").apply(select_random_rows)
        IDs2 = (Activity["Main_tile"] == "No") & (Activity[output] == "Inactive")
        Activity = Activity[~IDs2]

        input_fasta_data_A = input_fasta_data_A[~IDs2]

        
    Y = Activity[output]
    # Y = pd.get_dummies(Y)["Active"]
    
    # Print the frequency of each level in the column
    print(Y.value_counts())
    
    print(set)

    with open(set + '_seq' + '.fa','w') as f:
        for item in input_fasta_data_A:
            f.write(item + '\n')
            
    
    with open(set + '_exp_' + output + '.fa', 'w') as f:
        for item in Y:
            f.write(item + '\n')

    return  input_fasta_data_A, Y

if __name__ == '__main__':   
    fold = 'fold09'
    output = 'epidermis'
    set = 'Val'
    X_sequence, Y = prepare_input(fold, set, output)
    set = 'Train'
    X_sequence, Y = prepare_input(fold, set, output)
    set = 'Test'
    X_sequence, Y = prepare_input(fold, set, output)
    
    
    
