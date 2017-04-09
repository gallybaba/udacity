import sklearn
from sklearn.svm import SVC

def run_svc(std_train_data, std_train_label, std_test_data, std_test_label):
    model = SVC()
    model.fit(std_train_data, std_train_label)
    s = model.score(std_test_data, std_test_label)
    return s
