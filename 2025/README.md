# March Madness with Machine Learning (2025)

Welcome to my March Madness 2025 repository! This repository contains all the code and data I used to train a model to assist with the creation of my bracket for 2025 and entering the Kaggle [March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/).

## Table of Contents

- [Data Preprocessing](#data-preprocessing)
- [Logistic Regression](#logistic-regression)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)

## Data Preprocessing
For 2025 I have decided only to use [data](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data) provided by Kaggle to avoid having to handle any team spelling differences across datasets. Kaggle has very nicely assigned UIDs to each team to keep track of them throughout the seasons and without comparing strings.

For this year, most of the data will come from box score stats provided in [MRegularSeasonDetailedResults.csv](/data/men%20data/MRegularSeasonDetailedResults.csv):

Description from [Kaggle:](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data)
- **WFGM** - field goals made (by the winning team)
- **WFGA** - field goals attempted (by the winning team)
- **WFGM3** - three pointers made (by the winning team)
- **WFGA3** - three pointers attempted (by the winning team)
- **WFTM** - free throws made (by the winning team)
- **WFTA** - free throws attempted (by the winning team)
- **WOR** - offensive rebounds (pulled by the winning team)
- **WDR** - defensive rebounds (pulled by the winning team)
- **WAst** - assists (by the winning team)
- **WTO** - turnovers committed (by the winning team)
- **WStl** - steals (accomplished by the winning team)
- **WBlk** - blocks (accomplished by the winning team)
- **WPF** - personal fouls committed (by the winning team)

(And then the same set of stats from the perspective of the losing team: LFGM is the number of field goals made by the losing team, and so on up to LPF).

I began by splitting the data to focus on one team to track the number of wins and losses for each team, and the number of games played by that team.

So for each row in MRegularSeasonDetailedResults.csv, I split it into two rows, one for each team. Then I grouped the dataframe on the Season and TeamID, summing the results, ending up with the these columns:<br>

['Score', 'OppScore', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM',<br>
       'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'OppFGM',<br>
       'OppFGA', 'OppFGM3', 'OppFGA3', 'OppFTM', 'OppFTA', 'OppOR',<br>
       'OppDR', 'OppAst', 'OppTO', 'OppStl', 'OppBlk', 'OppPF', 'Wins',<br>
       'Losses', 'NumGames']

### Feature Creation
To get the features to feed to the model I started by getting a per game stat for each box score stat above excluding 'Wins', 'Losses', and 'NumGames'.

I then created the following additional features from the original box score stats that provides insights to how each team plays:
- **Points Ratio** - ratio of points scored to opponent's points scored.
- **Win/Loss Ratio** - self-explanatory.
- **Margin of Victory** - on average, how close is the score of each game.
- **Turnover Ratio** - ratio of TOs to OppTOs.
- **Scoring Efficiency** - ratio of made shots to attempts. (How well does a team shoot in general)
- **3-Point Efficiency** - ratio of 3-pointers made to attempted. (How well does a team shoot 3s)
- **3-Point Attempt Rate** - ratio of 3-pointers attempted to all shots attempted. (How reliant is a team on 3s)
- **Free Throw Efficiency** - ratio of free throws made to attempted. (How well does a team make free throws)
- **Free Throw Attempt Rate** - ratio of free throws attempted to all shots attempted. (How often a team gets to the line)
- **Opponent Free Throw Attempt Rate** - ratio of free throws attempted by opponents to all shots attempted by opponents. (How often a team send their opponent to the line)
- **Offensive Rebound Rate** - ratio of ORs to all rebounds that happen while they are on offense. (How well a team attacks the glass on offense)
- **Defensive Rebound Rate** - ratio of DRs to all rebounds that happen while they are on defense. (How well a team attacks the glass on defense)

## Logistic Regression
For this model I am using the [Scikit-Learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) framework.

## Hyperparameter Tuning
Here are the optimal parameters I have found using a grid search:

`C=1,
max_iter=100,
solver='saga',
tol=0.0001
`

## Results
Here are my current results combining men's and women's data into a single model:

**Logistic Regression:**

- Log Loss: 0.5749689958621383
- Brier Score: 0.1984564004495349

My target brier score is 0.155. Needless to say, I have some work to do.
