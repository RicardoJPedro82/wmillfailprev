from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class Trainer():
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.kwargs = kwargs

    def get_estimator(self, component):
        """return estimator"""
        if component == "Generator":
            model = KNeighborsClassifier()
        elif component == "Hydraulic":
            model = LogisticRegression()
        elif component == "Transformer":
            model = AdaBoostClassifier()
        elif component == "Generator Bearing":
            model = KNeighborsClassifier()
        elif component == "Gearbox":
            model = RandomForestClassifier()
        else:
            print('Invalid Component')

        return model

    def set_pipeline(self, component):
        """defines the pipeline as a class attribute"""
        preprocessing_pipe = Pipeline([('roll_avg', add_features()),
                            ('scaler', scale())])

        self.model_pipe = Pipeline([('preprocess', preprocessing_pipe),
                                    ('model', self.get_estimator(component))])


    def train(self, component):
        """set and train the pipeline"""
        self.set_pipeline()
        self.model_pipe.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return metrics"""
        y_pred = self.model_pipe.predict(X_test)
        savings, cf_numbers, met_cre_df, dias_primeiro_TP = metrics_create_df(X_test, y_test, y_pred, component, days=10)
        return savings


if __name__ == "__main__":
    # # get data
    # df = get_data()
    # # clean data
    # df = clean_data(df)
    # # set X and y
    # X = df.drop(columns='fare_amount')
    # y = df['fare_amount']
    # # hold out
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # # train
    # trainer = Trainer(X_train, y_train, estimator='RandomForest')
    # trainer.train()
    # # evaluate
    # rmse = trainer.evaluate(X_test, y_test)

    # print(rmse)
    # print('TODO')

