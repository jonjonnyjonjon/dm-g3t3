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
