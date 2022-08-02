from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


def create_model(df):
    train_X, test_X, train_y, test_y = train_test_split(df.drop('variety', axis=1),
                                                        df.variety,
                                                        train_size=.8,
                                                        stratify=df.variety,
                                                        random_state=123)

    rf = RandomForestClassifier()
    rf.fit(train_X, train_y)
    preds_train = rf.predict(train_X)
    preds_test = rf.predict(test_X)

    conf_matr_train = confusion_matrix(train_y, preds_train)
    conf_matr_test = confusion_matrix(test_y, preds_test)
    acc_train = accuracy_score(train_y, preds_train)
    acc_test = accuracy_score(test_y, preds_test)

    return rf, acc_train, acc_test, conf_matr_train, conf_matr_test


# if __name__ == "__main__":
#     create_model()