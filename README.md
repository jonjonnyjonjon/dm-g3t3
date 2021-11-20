# Libraries and resources used

#### Exploratory Data Analysis
- pandas
- numpy
- matplotlib.pyplot
- seaborn

#### Encoding
- sklearn.preprocessing.LabelEncoder
- sklearn.preprocessing.OneHotEncoder
- category_encoders.TargetEncoder

#### Feature Selection
- sklearn.feature_selection.chi2
- sklearn.feature_selection.SelectKBest
- sklearn.ensemble.RandomForestClassifier
- xgboost.XGBClassifier (download instructions can be found [here](https://xgboost.readthedocs.io/en/latest/install.html "here"))

#### Feature Scaling
- sklearn.preprocessing.MinMaxScaler

#### Models
- sklearn.ensemble.AdaBoostClassifier
- sklearn.tree.DecisionTreeClassifier
- sklearn.linear_model.LogisticRegression
- sklearn.neighbors.KNeighborsClassifier
- sklearn.ensemble.RandomForestClassifier
- sklearn.svm.SVC
- xgboost.XGBClassifier

#### Modelling tools
- imblearn.over_sampling.SMOTE
- imblearn.pipeline.Pipeline
- sklearn.model_selection
	- train_test_split, StratifiedKFold, cross_validate, GridSearchCV
- sklearn.compose.ColumnTransformer

#### Tensorflow and Keras libraries for Neural Network
- tensorflow.keras
	- models.Sequential
	- layers.Dense
	- utils.to_categorical
	- metrics.Recall, metrics.AUC, metrics.Precision
	- callbacks.EarlyStopping
	- optimizers.Adam
- tensorflow_addons.tfa

#### Evaluation metrics
- sklearn.metrics
	- accuracy_score, precision_score, recall_score, fbeta_score, roc_auc_score, make_scorer
