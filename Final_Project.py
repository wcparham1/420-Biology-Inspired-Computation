import numpy as np
import matplotlib as pyplot
import pandas as pd
from geneal.genetic_algorithms import ContinuousGenAlgSolver, BinaryGenAlgSolver
from geneal.applications.fitness_functions.continuous import fitness_functions_continuous


def merge(list1, list2):
     
    merged_list = tuple(zip(list1, list2))
    return merged_list

class TemplateChildClass(ContinuousGenAlgSolver, BinaryGenAlgSolver):
    def __init__(self, *args, **kwargs):
        BinaryGenAlgSolver.__init__(self, *args, **kwargs)
        ContinuousGenAlgSolver.__init__(self, *args, **kwargs)

    #a chromosome will be an individual player
    def fitness_function(self, chromosome):
        """
        Implements the logic that calculates the fitness
        measure of an individual.

        :param chromosome: chromosome of genes representing an individual
        :return: the fitness of the individual
        """
        pass

    def initialize_population(self):
        """
        Initializes the population of the problem

        :param pop_size: number of individuals in the population
        :param n_genes: number of genes representing the problem. In case of the binary
        solver, it represents the number of genes times the number of bits per gene
        :return: a numpy array with a randomized initialized population
        """
        pass

    def create_offspring(
        self, first_parent, sec_parent, crossover_pt, offspring_number
    ):
        """
        Creates an offspring from 2 parents. It uses the crossover point(s)
        to determine how to perform the crossover

        :param first_parent: first parent's chromosome
        :param sec_parent: second parent's chromosome
        :param crossover_pt: point(s) at which to perform the crossover
        :param offspring_number: whether it's the first or second offspring from a pair of parents.
        Important if there's different logic to be applied to each case.
        :return: the resulting offspring.
        """
        pass

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population according to a given user defined rule.

        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed. This number is 
        calculated according to mutation_rate, but can be adjusted as needed inside this function
        :return: the mutated population
        """
        pass


def main():
    
    #read in our data csv
    df = pd.read_csv('player_info/archive/NBA_Dataset.csv')
    season_2022_rows = df.loc[df['season']==2022]
    team_list = season_2022_rows['team_id'].unique()
    
    print(season_2022_rows.columns)
    # season_2022_rows = season_2022_rows.iloc[:, :-20]
    # season_2022_rows = season_2022_rows.iloc[:,2:]
    
    useful_stats = season_2022_rows[['fg_per_g', 'stl_per_g', 'blk_per_g']].copy()
    print('da stats: \n', useful_stats.head)
    #The important stats for what we are doing will be fg_per_g, stl_per_g, blk_per_g
    print('\n da old stuff: \n', season_2022_rows['fg_per_g'].head)
    # team_TOR = season_2022_rows.loc[season_2022_rows['team_id']=='TOR']

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
    for title in season_2022_rows.columns:
        mins.append(season_2022_rows[title].min())
        maxs.append(season_2022_rows[title].max())    
    
    min_maxs = list(merge(mins, maxs))
    
    #Now that we have it sorted by position we can use the genetic algorithm to start "growing" the best player
    #Look into the readme for instructions on how to further use this library.  It is solving *something*
    #But not solving the lineup issue right now

    # print("These are the Population sizes by position: ")
    # for i in range(0, len(players_by_position)):
    #     print('role: ', pos_list[i], 'pop_size: ', len(players_by_position[i]))

    solver = ContinuousGenAlgSolver(
        n_genes=55,
        fitness_function=fitness_functions_continuous(3),
        pop_size=106,
        plot_results=False,
    )


if __name__ == "__main__":
    main()