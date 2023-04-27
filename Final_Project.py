import numpy as np
import matplotlib as pyplot
import pandas as pd
from geneal.genetic_algorithms import ContinuousGenAlgSolver
from geneal.applications.fitness_functions.continuous import fitness_functions_continuous


def main():
    
    df = pd.read_csv('player_info/archive/NBA_Dataset.csv')

    print(len(df.columns))

    season_2022_rows = df.loc[df['season']==2022]

    team_list = season_2022_rows['team_id'].unique()

    team_TOR = season_2022_rows.loc[season_2022_rows['team_id']=='TOR']

    #iterate through each team contained in team_list and append each lineup to the rosters_2022 list
    rosters_2022 = []
    for team in team_list:
        rosters_2022.append(season_2022_rows.loc[season_2022_rows['team_id']==team])

    pos_list = season_2022_rows['pos'].unique()

    players_by_position = []
    for pos in pos_list:
        players_by_position.append(season_2022_rows.loc[season_2022_rows['pos']==pos])

        
    #Now that we have it sorted by position we can use the genetic algorithm to start "growing" the best player
    #Look into the readme for instructions on how to further use this library.  It is solving *something*
    #But not solving the lineup issue right now

    solver = ContinuousGenAlgSolver(
        n_genes=55,
        fitness_function=fitness_functions_continuous(3),
    )

    solver.solve()

if __name__ == "__main__":
    main()