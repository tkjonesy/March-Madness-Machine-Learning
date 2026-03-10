import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


'''
Function to merge two dataframes
param df1: first dataframe
param df2: second dataframe
return: a single dataframe of the merged dataframes 
'''
def mergeDataframes(df1, df2):
    df = pd.concat([df1, df2])
    return df


'''
Function to merge tournament data for mens and womens and crop the data to match reg detail
params df1: dataframe of mens tournament data
params df2: dataframe of womens tournament data
return: dataframe of combined and cropped tournament data
'''
def mergeTournamentData(df1, df2):
    df1 = df1[df1['Season'] >= 2003].reset_index(drop=True)
    df2 = df2[df2['Season'] >= 2010].reset_index(drop=True)
    
    df = pd.concat([df1, df2])
    return df


'''
Function to split regular season detailed results into dataframes focused on outcome for one team
param df: regular season data
return: a dataframe where each team from a single row in reg data has its own row
'''
def regularDetailsFocus(df):
    RegWinners = pd.DataFrame()
    RegLossers = pd.DataFrame()

    # Establish new columns for that includes stats for one team
    columns = ['Season', 'TeamID', 'Score', 'OppScore',
               'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
               'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'OppFGM', 'OppFGA',
               'OppFGM3', 'OppFGA3', 'OppFTM', 'OppFTA', 'OppOR', 'OppDR', 'OppAst', 'OppTO',
               'OppStl', 'OppBlk', 'OppPF']

    # Split winners from regular season
    RegWinners[columns] = df[['Season', 'WTeamID', 'WScore', 'LScore',
                              'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
                              'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA',
                              'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO',
                              'LStl', 'LBlk', 'LPF']]

    # Add wins and losses columns
    RegWinners['Wins'] = 1
    RegWinners['Losses'] = 0

    # Split lossers from regular season
    RegLossers[columns] = df[['Season', 'LTeamID', 'LScore', 'WScore',
                               'NumOT', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
                               'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'WFGM', 'WFGA',
                               'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO',
                               'WStl', 'WBlk', 'WPF']]

    # Add wins and losses columns
    RegLossers['Wins'] = 0
    RegLossers['Losses'] = 1

    # Combine all games into one dataframe
    AllRegDetail = pd.concat([RegWinners, RegLossers])
    return AllRegDetail


'''
Function to sum all the data for each season. Change the group dataframe to (season, teamID)
param df: dataframe where each team has its own row
return: dataframe of summed stats per season per team
'''
def sumStats(df):
    RegSeasonDetail = df.groupby(['Season', 'TeamID']).sum(numeric_only=True)
    RegSeasonDetail['NumGames'] = RegSeasonDetail['Wins'] + RegSeasonDetail['Losses']

    # Replace 0s in Losses with a small number to avoid dividing by 0 later
    RegSeasonDetail['Losses'] = RegSeasonDetail['Losses'].replace(0, 1e-5)
    return RegSeasonDetail


'''
Function to create the features from the regular season data
param df: dataframe of summed regular season data
return: dataframe of input features for model
'''
def createFeatures(df):
    RegSeasonFeatures = pd.DataFrame()

    # Create ___ per game stat for each boxscore stat
    stats = ['Score', 'OppScore', 'NumOT', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
             'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'OppFGA',
             'OppFGM3', 'OppFGA3', 'OppFTM', 'OppFTA', 'OppOR', 'OppDR', 'OppAst', 'OppTO',
             'OppStl', 'OppBlk', 'OppPF']

    for col in stats:
        RegSeasonFeatures[col + '_PerGame'] = df[col] / df['NumGames']

    # Create additional features
    RegSeasonFeatures['PointRatio'] = df['Score'] / df['OppScore']  # Points ratio
    RegSeasonFeatures['W/L'] = df['Wins'] / df['Losses']  # Win/Loss ratio
    RegSeasonFeatures['MOV'] = (df['Score'] - df['OppScore']) / df['NumGames']  # Margin of victory
    RegSeasonFeatures['TORatio'] = RegSeasonFeatures['TO_PerGame'] / RegSeasonFeatures['OppTO_PerGame']  # Turnover ratio
    RegSeasonFeatures['FGM%'] = df['FGM'] / df['FGA']  # Scoring efficiency
    RegSeasonFeatures['FG3%M'] = df['FGM3'] / df['FGA3']  # 3-Point efficiency
    RegSeasonFeatures['FGA3%'] = df['FGA3'] / df['FGA']  # 3-Point attempt rate
    RegSeasonFeatures['FTM%'] = df['FTM'] / df['FTA']  # Free throw makes %
    RegSeasonFeatures['OppFTM%'] = df['OppFTM'] / df['OppFTA']  # Opponent free throw makes %
    RegSeasonFeatures['FTA%'] = df['FTA'] / df['FGA']  # Free throw attempt rate
    RegSeasonFeatures['OppFTA%'] = df['OppFTA'] / df['OppFGA']  # Opponent free throw attempt rate
    RegSeasonFeatures['OR%'] = df['OR'] / (df['OR'] + df['OppDR'])  # Offensive rebound %
    RegSeasonFeatures['DR%'] = df['DR'] / (df['DR'] + df['OppOR'])  # Defensive rebound %
    return RegSeasonFeatures


'''
Function to split tournament data to match the format of input features
param df: dataframe of compact tournament data
return: dataframe of tournament data ready to merge with features
'''
def splitHistTournamentData(df):
    WTourney = pd.DataFrame()
    LTourney = pd.DataFrame()
    
    # Setup separate dataframes for tournament data and add target feature
    WTourney[['Season', 'Team1', 'Team2']] = df[['Season', 'WTeamID', 'LTeamID']]
    WTourney['Result'] = 1

    LTourney[['Season', 'Team1', 'Team2']] = df[['Season', 'LTeamID', 'WTeamID']]
    LTourney['Result'] = 0

    # Join individual together
    TourneyInput = pd.concat([WTourney, LTourney])
    return TourneyInput


'''
Function to calculate difference in teams input features for historical tournament matchups for LR
param df1: dataframe of tournament data
param df2: dataframe of input features
return: dataframe of differences for training
'''
def calculateDifferenceHistTourn(df1, df2):
    # Merge two team stats from RegSeasonFeatures
    TourneyFinal = df1.merge(df2, left_on=['Season', 'Team1'], right_index=True, suffixes=('', '_T1'))
    TourneyFinal = TourneyFinal.merge(df2, left_on=['Season', 'Team2'], right_index=True, suffixes=('_T1', '_T2'))

    # Drop columns that are not needed
    TourneyFinal.drop(columns=['Season', 'Team1', 'Team2'], inplace=True)

    # Calculate the differences (Team1 - Team2) for the features for input to logistic regression
    featureCols = [col for col in df2 if col not in ['Season', 'TeamID']]
    for col in featureCols:
        TourneyFinal[col + '_Diff'] = TourneyFinal[col + '_T1'] - TourneyFinal[col + '_T2']

    # Drop all _T1 and _T2, keep only _Diff and Result
    TourneyFinal = TourneyFinal[[col + '_Diff' for col in featureCols] + ['Result']]
    return TourneyFinal


'''
Function to handle setting up train/test split
param df: dataframe of data
return: 4 dataframes containing xtrain, xtest, ytrain, and ytest
'''
def trainTestSplit(df):
    scaler = StandardScaler()

    # Setup input and target
    X = df.drop(columns=['Result'])
    y = df['Result']

    # Split into train and test sets
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit scaler on training data and transform both train and test sets
    XTrain = scaler.fit_transform(XTrain)
    XTest = scaler.transform(XTest)
    return XTrain, XTest, yTrain, yTest


'''
Function to take example submission dataframe seperate one column into 3.
param df: dataframe with single column 'ID'. Expects '2025_1101_1102' format
return: dataframe with three columns ['Season', 'Team1', 'Team2']
'''
def splitSubmission(df):
    splitData = df['ID'].str.split('_', expand=True)
    df[['Season', 'Team1', 'Team2']] = splitData.astype(int)
    df.drop(columns=['ID'], inplace=True)
    return df


'''
Function to calculate difference in teams input features for final prediction
param df1: dataframe of all team pairings for 2025
param df2: dataframe of input features
return: dataframe of differences for training
'''
def calculateDifferenceFinal(df1, df2):
    # Merge two team stats from RegSeasonFeatures
    TourneyFinal = df1.merge(df2, left_on=['Season', 'Team1'], right_index=True, suffixes=('', '_T1'))
    TourneyFinal = TourneyFinal.merge(df2, left_on=['Season', 'Team2'], right_index=True, suffixes=('_T1', '_T2'))

    # Drop columns that are not needed
    TourneyFinal.drop(columns=['Season', 'Team1', 'Team2'], inplace=True)

    # Calculate the differences (Team1 - Team2) for the features for input to logistic regression
    featureCols = [col for col in df2 if col not in ['Season', 'TeamID']]
    for col in featureCols:
        TourneyFinal[col + '_Diff'] = TourneyFinal[col + '_T1'] - TourneyFinal[col + '_T2']

    # Drop all _T1 and _T2, keep only _Diff
    TourneyFinal = TourneyFinal[[col + '_Diff' for col in featureCols]]

    # Scale final input
    scaler = StandardScaler()
    TourneyFinal = scaler.fit_transform(TourneyFinal)
    return TourneyFinal


'''
Function to combine predictions to match the format of submission file for kaggle
param df: dataframe with columns ['Season', 'Team1', 'Team2']
param yProba: array of predicted probabilities
return: dataframe with 'ID' and 'Pred' columns
'''
def combinePredictions(df, yProba):
    # Reconstruct ID column
    df['ID'] = df['Season'].astype(str) + '_' + df['Team1'].astype(str) + '_' + df['Team2'].astype(str)

    # Add prediciton column
    df['Pred'] = yProba[:, 1]
    return df[['ID', 'Pred']]


'''
Function to add team names to prediction file to make a bracket with
param df: dataframe with columns ['Season', 'Team1', 'Team2']
param yProba: array of predicted probabilities
param mNames: dataframe with columns ['TeamNameSpelling', 'TeamID']
param wNames: dataframe with columns ['TeamNameSpelling', 'TeamID']
'''
def addPredsWithNames(df, yProba, mNames, wNames):
    # Create mapping for team names and IDs
    teamNames = pd.concat([mNames, wNames])
    teamNames = teamNames.drop_duplicates(subset='TeamID', keep='first')
    teamMap = teamNames.set_index('TeamID')['TeamNameSpelling']

    # Add predicted probabilities to the DataFrame
    df['Pred'] = yProba[:, 1]

    # Map team IDs to team names
    df['Team1Name'] = df['Team1'].map(teamMap)
    df['Team2Name'] = df['Team2'].map(teamMap)
    return df[['Team1', 'Team1Name', 'Team2', 'Team2Name', 'Pred']]