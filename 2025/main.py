import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from functions import mergeDataframes
from functions import mergeTournamentData
from functions import regularDetailsFocus
from functions import sumStats
from functions import createFeatures
from functions import splitHistTournamentData
from functions import calculateDifferenceHistTourn
from functions import trainTestSplit
from functions import splitSubmission
from functions import calculateDifferenceFinal
from functions import combinePredictions
from functions import addPredsWithNames

if __name__ == "__main__":
    # Competition input
    finalPairings = pd.read_csv('../submissions/SampleSubmissionStage2.csv')
    finalPairings.drop(columns=['Pred'], inplace=True)
    finalPairings = splitSubmission(finalPairings)

    # Mens data import
    mRegDetail = pd.read_csv('../data/men data/MRegularSeasonDetailedResults.csv')
    mTournCompact = pd.read_csv('../data/men data/MNCAATourneyCompactResults.csv')
    mNames = pd.read_csv('../data/men data/MTeamSpellings.csv')

    # Womens data import
    wRegDetail = pd.read_csv('../data/women data/WRegularSeasonDetailedResults.csv')
    wTournCompact = pd.read_csv('../data/women data/WNCAATourneyCompactResults.csv')
    wNames = pd.read_csv('../data/women data/WTeamSpellings.csv')

    # Combined data
    regDetail = mergeDataframes(mRegDetail, wRegDetail)
    compactTourn = mergeTournamentData(mTournCompact, wTournCompact)
    names = mergeDataframes(mNames, wNames)

    # Split regular season detailed results into dataframes focused on outcome for one team
    AllRegDetail = regularDetailsFocus(regDetail)

    # Sum regular season stats per season
    RegSeasonDetail = sumStats(AllRegDetail)

    # Create input features
    RegSeasonFeatures = createFeatures(RegSeasonDetail)

    # Create correlation matrix and locate highly correlated features
    corrs = round(RegSeasonFeatures.corr(), 2).abs()
    upper = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(bool))

    # Find pairs with correlation >= 0.7 and include the correlation score
    highCorrPairs = [(col1, col2, upper.loc[col2, col1])
                     for col1 in upper.columns
                     for col2 in upper.index
                     if upper.loc[col2, col1] >= 0.7]

    # Display the results
    for pair in highCorrPairs:
        print(pair)

    # Handle historical tournament data
    TourneyInput = splitHistTournamentData(compactTourn)

    # Calculate historical differences for training
    histData = calculateDifferenceHistTourn(TourneyInput, RegSeasonFeatures)

    # Setup train test split
    XTrain, XTest, yTrain, yTest = trainTestSplit(histData)

    # Setup and train Logistic Regression model
    model = LogisticRegression(
        random_state=42,
        C=1,
        max_iter=1000,
        solver='saga',
        tol=0.0001
    )
    model.fit(XTrain, yTrain)

    # Get model predictions
    yPred = model.predict(XTest)
    yPredProba = model.predict_proba(XTest)

    # Compute Metrics log loss and brier score
    logLoss = log_loss(yTest, yPredProba)
    print(f'Log Loss: {logLoss}')
    yPredProbaClass1 = yPredProba[:, 1]
    brierScore = brier_score_loss(yTest, yPredProbaClass1)
    print(f'Brier Score: {brierScore}')

    # Setup for final prediction
    finalInput = calculateDifferenceFinal(finalPairings, RegSeasonFeatures)
    yFinalProba = model.predict_proba(finalInput)
    submissionFinal = combinePredictions(finalPairings, yFinalProba)
    submissionFinal.to_csv('submissions/submission1.csv', index=False)

    # Create output to use for bracket with team names
    bracketPredicitons = addPredsWithNames(finalPairings, yFinalProba, mNames, wNames)
    bracketPredicitons.to_csv('submissions/bracketPredictions.csv', index=False)