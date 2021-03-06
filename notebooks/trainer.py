import pandas as pd
from Dataset_transf import *
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from termcolor import colored

class Trainer():
    def __init__(self, x_train, y_train, x_test, y_test, **kwargs):

        self.pipeline = None
        self.kwargs = kwargs
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_estimator(self):
        """return estimator"""
        component = self.kwargs.get("component")
        if component == "df_generator":
            model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                        metric_params=None, n_jobs=-1, n_neighbors=5,
                                        p=2, weights='uniform')
        elif component == "df_hydraulic":
            model = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                                        intercept_scaling=1, max_iter=1000, multi_class='warn',
                                        n_jobs=-1, penalty='l2', solver='lbfgs',
                                        tol=0.0001, verbose=0, warm_start=False)
        elif component == "df_transformer":
            model = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                                        learning_rate=1.0, n_estimators=200)
        elif component == "df_gen_bear":
            model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                        metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                                        weights='distance')
        elif component == "df_gearbox":
            model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                                        min_impurity_decrease=0.0, min_impurity_split=None,
                                        min_samples_leaf=1, min_samples_split=2,
                                        min_weight_fraction_leaf=0.0, n_estimators=750, n_jobs=-1,
                                        oob_score=False, verbose=0, warm_start=False)
        else:
            print('Invalid Component')

        return model

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        component = self.kwargs.get("component")
        preprocessing_pipe = Pipeline([('roll_avg', add_features(self.x_train)),
                                        ('scaler', StandardScaler())])

        self.model_pipe = Pipeline([('preprocess', preprocessing_pipe),
                                    ('model', self.get_estimator(component))])


    def train(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.model_pipe.fit(self.x_train, self.y_train)

    def evaluate(self):
        """evaluates the pipeline on df_test and return metrics"""
        component = self.kwargs.get("component")
        y_pred = self.model_pipe.predict(self.x_test)
        savings, cf_numbers, met_cre_df, dias_primeiro_TP = metrics_create_df(self.x_test, self.y_test, self.y_pred, component, days=10)
        return savings

    def save_model(self):
        """Save the model into a .joblib models folder"""
        component = self.kwargs.get("component")
        joblib.dump(self.pipeline, f'model_{component}.joblib')
        print(colored(f'model_{component}.joblib saved locally', "green"))


if __name__ == "__main__":
    print("Olá")
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set x and y
    # hold out
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # train
    #for component in components:
        #trainer = Trainer(x_train[component], y_train[component])
        # print(colored("############  Training model   ############", "red"))
        # trainer.train()
        # print(colored("############  Evaluating model ############", "blue"))
        # trainer.evaluate()
        # print(colored("############   Saving model    ############", "green"))
        # trainer.save_model()
