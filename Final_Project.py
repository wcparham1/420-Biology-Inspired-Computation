import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from geneal.genetic_algorithms import ContinuousGenAlgSolver, BinaryGenAlgSolver
from geneal.applications.fitness_functions.continuous import fitness_functions_continuous

#When you want to look at a new position set this variable to a value between 0-4
POSITION = 4


#This function rescales values between two previous min and maxes.
#Use it when converting a player's statistics from the evolutionary algorithm to the csv.
def rescale_value(old_max, old_min, new_max, new_min, old_value):
    return (old_max - old_min) * (old_value - new_min) / (new_max - new_min) + old_min
    
#This function merges two lists together.  Use it when creating the old_min_max list.
#Perhaps get rid of this function later.
def merge(list1, list2):
    merged_list = tuple(zip(list1, list2))
    return merged_list

#This is the fitness function we are using when evolving players with the evolutionary algorithm.
#We can change this lambda function inside to weight different attributes accordingly.
def my_fitness_function():    
    ret_val = lambda chromosome: (1.5*chromosome[0] + 1.0*chromosome[1] + 2.0*chromosome[2])
    return ret_val

#This function performs min_max normalization between two bounds.  We need it to reconvert our evolutionary player's fitness value to
#the values obtained in the csv.
def perform_min_max_normalization(minv, maxv, prec, vals):
    unique_values = int(((maxv-minv) / prec) + 1)
    return ((vals - minv) / (maxv + prec - minv) * unique_values).astype(int)

#main...
def main():
    
    #read in our data csv
    df = pd.read_csv('player_info/archive/NBA_Dataset.csv')
    season_2022_rows = df.loc[df['season']==2022]
    team_list = season_2022_rows['team_id'].unique()
    
    useful_stats = season_2022_rows[['fg_per_g', 'stl_per_g', 'blk_per_g']].copy()
    
    # The important stats for what we are doing will be fg_per_g, stl_per_g, blk_per_g
    # We take each position in order to sort by position when generating the best player for each role

    pos_list = season_2022_rows['pos'].unique()

    players_by_position = []
    for pos in pos_list:
        players_by_position.append(season_2022_rows.loc[season_2022_rows['pos']==pos])

    stats_by_position = []
    for frame in players_by_position:
        stats_by_position.append(frame[['fg_per_g', 'stl_per_g', 'blk_per_g']].copy())
    
    names_by_position = []
    for frame in players_by_position:
        names_by_position.append(frame[['player']].copy())
        
    #look at each position and find the min/max of each statistic.
    mins = []
    maxs = []
    for col_name in stats_by_position[POSITION].columns:                                        #CHANGE THIS LINE WHEN LOOKING AT DIFFERENT POSITION
        mins.append(stats_by_position[POSITION][col_name].min())                                #CHANGE THIS LINE WHEN LOOKING AT DIFFERENT POSITION
        maxs.append(stats_by_position[POSITION][col_name].max())                                #CHANGE THIS LINE WHEN LOOKING AT DIFFERENT POSITION
    
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
        max_gen=100,
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
    
    #1
    #Currently the scaled member is the most fit individual rescaled to match the statistics csv we are 
    #using to create the individual.  The reason we have to rescale is because we cannot set bounds on
    #the evolutionary algorithm.  Currently it is evolving an individual that matches all the highest stats
    #we give it.  I.E. the upper bound in the variable_limits parameter in the solver.
    
    #2
    #Now we need to write an algorithm that matches this fit individual with a player in the csv.
    rescaled_member = [0,0,0]    
    rescaled_member[0] = rescale_value(
                            old_min=min_maxs_old[0][0],
                            old_max=min_maxs_old[0][1],  
                            new_min=new_mins[0], 
                            new_max=new_maxs[0], 
                            old_value=solver.best_individual_[0]
                            )
    rescaled_member[1] = rescale_value(
                            old_min=min_maxs_old[1][0],
                            old_max=min_maxs_old[1][1],  
                            new_min=new_mins[1], 
                            new_max=new_maxs[1], 
                            old_value=solver.best_individual_[1]
                            )
    rescaled_member[2] = rescale_value(
                            old_min=min_maxs_old[2][0],
                            old_max=min_maxs_old[2][1],  
                            new_min=new_mins[2], 
                            new_max=new_maxs[2], 
                            old_value=solver.best_individual_[2]
                            )
    
    #round the value of our newly evolved player and match to a value in the array
    print('rescaled member statistics: \n', rescaled_member)
    row_sums = stats_by_position[POSITION].sum(axis='columns')                                    #CHANGE THIS LINE WHEN LOOKING AT DIFFERENT POSITION        
    scaled_mem_sum = round(sum(rescaled_member), 1)
    
    #print('this is the sum of each row:', row_sums)
    print('this is the sum of rescaled member:', scaled_mem_sum)
    
    row_sums = list(row_sums)
    matched_val = 0
    scaling = 0
    matched_index = 0
    while(matched_val == 0):
        for row in range(0, len(row_sums)):
            if(row_sums[row] <= scaled_mem_sum + scaling and row_sums[row] >= scaled_mem_sum - scaling):
                matched_val = row_sums[row]
                matched_index = row
                print('perfect match found at: ', row)
                break
        print('no match found increasing the range to equality')
        scaling += 1
    
    print('matched index: ', matched_index)
    player_selected = names_by_position[POSITION].iloc[[matched_index]]                            #CHANGE THIS LINE WHEN LOOKING AT DIFFERENT POSITION
    
    print('this the current position:', pos_list[POSITION])
    print(player_selected, matched_val)

#CREATE A GRAPH DEPICTING AVERAGE SCORES IN EACH CATEGORY VS WHAT OUR PLAYER DOES IN EACH CATEGORY

#matched player indexes
# anthony davis 23
# lebron james 50
# devin booker 8
# kyrie irving 42
# jayson tatum 103 SF
    best_stats = []
    all_stats = []
    
    print(players_by_position[0].iloc[[23]][['fg_per_g', 'stl_per_g', 'blk_per_g']])
    
    best_stats.append(players_by_position[0].iloc[[23]][['player','fg_per_g', 'stl_per_g', 'blk_per_g']])
    best_stats.append(players_by_position[1].iloc[[50]][['player','fg_per_g', 'stl_per_g', 'blk_per_g']])
    best_stats.append(players_by_position[2].iloc[[8]][['player','fg_per_g', 'stl_per_g', 'blk_per_g']])
    best_stats.append(players_by_position[3].iloc[[42]][['player','fg_per_g', 'stl_per_g', 'blk_per_g']])
    best_stats.append(players_by_position[4].iloc[[103]][['player','fg_per_g', 'stl_per_g', 'blk_per_g']])
    
    players_by_position[0] = players_by_position[0][players_by_position[0].player != 'Anthony Davis']
    players_by_position[1] = players_by_position[1][players_by_position[1].player != 'LeBron James']
    players_by_position[2] = players_by_position[2][players_by_position[2].player != 'Devon Booker']
    players_by_position[3] = players_by_position[3][players_by_position[3].player != 'Kyrie Irving']
    players_by_position[4] = players_by_position[4][players_by_position[4].player != 'Jayson Tatum']
    
    
    all_stats.append(players_by_position[0][['player','fg_per_g', 'stl_per_g', 'blk_per_g']])
    all_stats.append(players_by_position[1][['player','fg_per_g', 'stl_per_g', 'blk_per_g']])
    all_stats.append(players_by_position[2][['player','fg_per_g', 'stl_per_g', 'blk_per_g']])
    all_stats.append(players_by_position[3][['player','fg_per_g', 'stl_per_g', 'blk_per_g']])
    all_stats.append(players_by_position[4][['player','fg_per_g', 'stl_per_g', 'blk_per_g']])
    
    all_players = pd.concat(all_stats)
    all_players = all_players[['fg_per_g', 'stl_per_g', 'blk_per_g']]
    avg_means = all_players.mean(axis=0)
    
    test = pd.concat(best_stats)
    test = test[['fg_per_g', 'stl_per_g', 'blk_per_g']]
    best_means = test.mean(axis=0)
    
    #best means contains the means from our best players
    #average means contains the means from all players besides the best
    print(best_means)
    print(avg_means)
    
    data = {'our_fg_per_g': best_means[0], 'our_stl_per_g': best_means[1], 'our_blk_per_g': best_means[2],
            'avg_fg_per_g': avg_means[0], 'avg_stl_per_g': avg_means[1], 'avg_blk_per_g': avg_means[2]}
    
    labels = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize= (10,5))

    plt.bar(labels, values, color = 'orange', width = 0.4)
    
    plt.xlabel('average respective team')
    plt.ylabel('average score for both teams')
    plt.title("Ideal Constructed Team vs. Average Team")
    plt.show()
if __name__ == "__main__":
    main()
    