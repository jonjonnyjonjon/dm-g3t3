# Instructions to run the notebooks
You will need Anaconda and Jupyter Notebook installed on your laptop. Using the Jupyter Notebook extension in Visual Studio Code is also possible.

To run each notebook, `Kernel > Restart & Run All`.

#### File Directory outline
1. 01 EDA and data preprocessing
2. 02 Base Models
3. 03 Hyperparameter Tuning
4. 04 Compilation of results
5. dataset

##### 01 Exploratory Data Analysis (EDA) and data preprocessing
- eda.ipynb ⟶ various exploration done on the dataset
- data-preprocesing.ipynb ⟶ the dataset is preprocessed here and exported into train.csv and test.csv for training models in later parts

##### 02 Base Models
For each notebook in this folder, 3 variations of models were trained on the train.csv with cross validation in place. The results can also be viewed in these notebooks.

##### 03 Hyperparameter Tuning
All notebooks in this folder were ran on Kaggle due to the lack of computational resources on local laptops. However, the codes were manipulated in hopes to allow for local execution.

##### 04 Compilation of results
The best parameters of each selected model is consolidated here and the test set (test.csv) is used to evaluate the model's performance. The feature importances graph is also plotted here for the chosen model.

# Libraries and resources used

#### Overall
- pandas
- numpy
- matplotlib.pyplot
- seaborn
- sklearn
- [category_encoders](https://contrib.scikit-learn.org/category_encoders/ "category_encoders")
- [xgboost](https://xgboost.readthedocs.io/en/latest/install.html "xgboost")
- [tensorflow](https://www.tensorflow.org/install "tensorflow")
- [imblearn](https://pypi.org/project/imbalanced-learn/ "imblearn")

#### Specific methods
- sklearn.preprocessing.LabelEncoder
- sklearn.preprocessing.OneHotEncoder
- category_encoders.TargetEncoder
- sklearn.feature_selection.chi2
- sklearn.feature_selection.SelectKBest
- sklearn.preprocessing.MinMaxScaler
- sklearn.ensemble.AdaBoostClassifier
- sklearn.tree.DecisionTreeClassifier
- sklearn.linear_model.LogisticRegression
- sklearn.neighbors.KNeighborsClassifier
- sklearn.ensemble.RandomForestClassifier
- sklearn.svm.SVC
- xgboost.XGBClassifier 
- imblearn.over_sampling.SMOTE
- imblearn.pipeline.Pipeline
- sklearn.model_selection
	- train_test_split, StratifiedKFold, cross_validate, GridSearchCV
- sklearn.compose.ColumnTransformer
- tensorflow.keras
	- models.Sequential
	- layers.Dense
	- utils.to_categorical
	- metrics.Recall, metrics.AUC, metrics.Precision
	- callbacks.EarlyStopping
	- optimizers.Adam
- tensorflow_addons.tfa
- sklearn.metrics
	- accuracy_score, precision_score, recall_score, fbeta_score, roc_auc_score, make_scorer
