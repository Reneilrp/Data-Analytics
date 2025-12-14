==========================================================================
FLOOD RISK ASSESSMENT & PREDICTION SYSTEM (GROUP 6)
==========================================================================

PROJECT OVERVIEW:
This project uses Machine Learning (Random Forest, SVM, and Logistic Regression) 
to analyze flood risk in Metro Manila. It identifies key drivers of flooding 
(Rainfall, Water Level, Soil Moisture) and predicts flood events with high accuracy.

==========================================================================
FILE DESCRIPTIONS (WHAT EACH FILE DOES):
==========================================================================

1. train_and_save.py
   - PURPOSE: This is the setup script. It loads the raw data, balances it 
     using SMOTE (to fix data imbalance), and trains the three machine learning models.
   - OUTPUT: It creates three model files (.pkl) and locks away a "Test Set" 
     (test_features.csv and test_labels.csv) that the models have never seen before.

2. load_and-test.py
   - PURPOSE: This script acts as the "Grader". It loads the saved models and 
     tests them against the locked Test Set to see how well they perform.
   - OUTPUT: Prints the Accuracy, Recall, Precision, and F1-Score for each model 
     in the terminal.

3. visualization_result.py
   - PURPOSE: Generates the visual charts for the research paper.
   - OUTPUT: Saves two images:
     (a) "Final_Confusion_Matrices.png" (shows how many floods were caught vs missed)
     (b) "Final_Feature_Importance.png" (shows that Rainfall is the #1 driver)

4. app.py
   - PURPOSE: A simulation tool for demonstration. You can manually type in 
     Rainfall or Water Level values inside the code to see if the system predicts 
     a "FLOOD WARNING" or "SAFE" status.
   - OUTPUT: Prints the prediction result in the terminal.

5. Flood_Datasets.csv
   - PURPOSE: The historical data source used to train the system.

==========================================================================
HOW TO RUN THE PROJECT (STEP-BY-STEP):
==========================================================================

STEP 1: INSTALL LIBRARIES
Open your terminal or command prompt and run:
   pip install pandas scikit-learn seaborn matplotlib imbalanced-learn joblib

STEP 2: TRAIN THE MODELS
Run the training script first to create the model files.
   Command: python train_and_save.py
   
   (Wait until you see "Step 1 Complete" and the .pkl files appear in the folder)

STEP 3: CHECK ACCURACY SCORES
Run the testing script to see the performance metrics.
   Command: python load_and-test.py

STEP 4: GENERATE CHARTS
Run the visualization script to create the graphs for the report.
   Command: python visualization_result.py

STEP 5: RUN THE DEMO
Run the app script to test specific scenarios.
   Command: python app.py

==========================================================================