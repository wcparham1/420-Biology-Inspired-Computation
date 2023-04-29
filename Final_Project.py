import numpy as np
import matplotlib as pyplot
import pandas as pd
from geneal.genetic_algorithms import ContinuousGenAlgSolver, BinaryGenAlgSolver
from geneal.applications.fitness_functions.continuous import fitness_functions_continuous

def rescale_value(old_max, old_min, new_max, new_min, old_value):
    old_range=(old_max - old_min)
    new_range=(new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value

def merge(list1, list2):
     
    merged_list = tuple(zip(list1, list2))
    return merged_list

def my_fitness_function():    
    ret_val = lambda chromosome: (1.5*chromosome[0] + 1.0*chromosome[1] + 2.0*chromosome[2])
    return ret_val


def perform_min_max_normalization(minv, maxv, prec, vals):
    unique_values = int(((maxv-minv) / prec) + 1)
    return ((vals - minv) / (maxv + prec - minv) * unique_values).astype(int)

def main():
    
    #read in our data csv
    df = pd.read_csv('player_info/archive/NBA_Dataset.csv')
    season_2022_rows = df.loc[df['season']==2022]
    team_list = season_2022_rows['team_id'].unique()
    
    useful_stats = season_2022_rows[['fg_per_g', 'stl_per_g', 'blk_per_g']].copy()
    
    # The important stats for what we are doing will be fg_per_g, stl_per_g, blk_per_g

    #iterate through each team contained in team_list and append each lineup to the rosters_2022 list
    rosters_2022 = []
    for team in team_list:
        rosters_2022.append(season_2022_rows.loc[season_2022_rows['team_id']==team])

    pos_list = season_2022_rows['pos'].unique()

    players_by_position = []
    for pos in pos_list:
        players_by_position.append(season_2022_rows.loc[season_2022_rows['pos']==pos])

    #look at each position and find the min/max of each statistic.
    mins = []
    maxs = []
    for col_name in useful_stats.columns:
        mins.append(useful_stats[col_name].min())
        maxs.append(useful_stats[col_name].max())    
    
    min_maxs_old = list(merge(mins, maxs))
    
    #Now that we have it sorted by position we can use the genetic algorithm to start "growing" the best player
    #Look into the readme for instructions on how to further use this library.  It is solving *something*
    #But not solving the lineup issue right now

    solver = ContinuousGenAlgSolver(
        n_genes=3,
        fitness_function=my_fitness_function(),
        pop_size=100,
        plot_results=False,
        variables_limits=min_maxs_old,
        random_state=42,
        verbose=False,
    )
    solver.solve()
        
    prev_fg_per_g = []
    prev_stl_per_g = []
    prev_blk_per_g = []
    for member in solver.population_:
        prev_fg_per_g.append(member[0])
        prev_stl_per_g.append(member[1])
        prev_blk_per_g.append(member[2])

    new_mins = []
    new_mins.append(min(prev_fg_per_g))
    new_mins.append(min(prev_stl_per_g))
    new_mins.append(min(prev_blk_per_g))
    
    new_maxs = []
    new_maxs.append(max(prev_fg_per_g))
    new_maxs.append(max(prev_stl_per_g))
    new_maxs.append(max(prev_blk_per_g))
    
    scaled_member = [0,0,0]
    
    scaled_member[0] = rescale_value(
                            old_min=min_maxs_old[0][0],
                            old_max=min_maxs_old[0][1],  
                            new_min=new_mins[0], 
                            new_max=new_maxs[0], 
                            old_value=solver.best_individual_[0]
                            )
    
    print('scaled_member[0]:', scaled_member[0])
    print(min_maxs_old[0][0], min_maxs_old[0][1], new_mins[0], new_maxs[0], solver.best_individual_[0])
    
if __name__ == "__main__":
    main()
    
    # y_interp = scipy.interpolate.interp1d()
    
    # solver.solve()
    #print('this is best individual: ', solver.best_individual_)
    # best_candidate = np.asarray(solver.best_individual_)
    # best_candidate[0] = np.reshape(best_candidate[0], (-1,1))
    # print('this is best candidate [0]', best_candidate[0])
    
    
    # scaled_candidate = [0,0,0]
    # scaled_candidate[0] = scaler.transform(best_candidate[0])
    # print(scaled_candidate[0])
    # scaled_candidate[0] = perform_min_max_normalization(minv = min_maxs[0][0], maxv=min_maxs[0][1], prec=.001, vals=best_candidate[0])
    
    # print(scaled_candidate[0])