import pandas as pd
import os

def write_to_txt(path, data):
    f = open(path,'w')
    i = 0
    while i < len(data):
        f.write(str(data[i]) + '\n')
        i = i + 1
    f.close()

def population_remove_flank(population) : 
    return_population = []
    for i in range(len(population)): 
        return_population= return_population + [(population[i][17:-13])]
    return return_population

if __name__ == '__main__':
    df = pd.read_csv('./aviv_evolution/Random_testdata_complex_media.txt', sep ='\t' )
    population_current = df.iloc[:,0].values
    population_current_fitness = df.iloc[:,1].values


    for i in range(0,len(population_current)) :
        if (len(population_current[i]) > 110) :
            population_current[i] = population_current[i][-110:]
        if (len(population_current[i]) < 110) : 
            while (len(population_current[i]) < 110) :
                population_current[i] = 'N'+population_current[i]
    
    population_current = population_remove_flank(population_current)
    
    seqs = population_current
    seqs = [item.upper() for item in seqs]
    expr = population_current_fitness
    
    save_seqs_file = "./aviv_evolution/Random_testdata_complex_media_seq.txt"
    save_expr_file = "./aviv_evolution/Random_testdata_complex_media_exp.txt"
    write_to_txt(save_seqs_file, seqs)
    write_to_txt(save_expr_file, expr)
    