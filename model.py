from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def decision_tree_train(X_train, y_train, selected_features, target, d = 10, m=1, print_results = True):
    """Fits a Decision Tree Classifier to train data and outputs the classification report and the classifier (clf) object.
    Takes in as arguments the split train data, a list of selected features, a string of the target name, a max_depth value (d), and min_sample_leaf value (m)
    """
  
    clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf = m, random_state=123)
    clf = clf.fit(X_train, y_train)
    accuracy = clf.score(X_train, y_train)
    y_pred = clf.predict(X_train)
    class_report = classification_report(y_train, y_pred,output_dict=True)
    
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        
    fp_rate = fp/(fp+tn)
    fn_rate = fn/(fn+tp)
    tp_rate = tp/(tp+fn)
    tn_rate = tn/(fp+tn)
    if print_results:
        print(f"TRAINING RESULTS: {type(clf).__name__}")
        print(f"Using features: {X_train.columns.to_list()}")
        print(f"Depth of {clf.max_depth}")
        print(f"Min Sample Leaf of {clf.min_samples_leaf}")
        print("----------------")
        print(f"Accuracy score on training set is: {accuracy:.2f}")
        print(classification_report(y_train, y_pred))


        print(f"False positive rate: {fp/(fp+tn):.2%}")
        print(f"False negative rate: {fn/(fn+tp):.2%}")
        print(f"True positive rate: {tp/(tp+fn):.2%}")
        print(f"True negative rate: {tn/(fp+tn):.2%}")
        print("----------------")
    
    train_report = {'d':clf.max_depth, 
                    'm':clf.min_samples_leaf,
                    'accuracy':accuracy, 
                    'precision':class_report['1']['precision'], 
                    'recall':class_report['1']['recall'],
                   'fp_rate':fp_rate,
                   'fn_rate':fn_rate,
                   'tp_rate':tp_rate,
                   'tn_rate':tn_rate}
    
    return clf, train_report

def classifier_validate(X_validate, y_validate, clf, print_results=True):
    """ Evaluates decision tree and random forest classifier models on validate (test) data. Takes as arguments the split validate data as well as the classifier object (clf) generated in the train function. Outputs the classification report with the results."""
    # d = clf.max_depth
    accuracy = clf.score(X_validate, y_validate)
    # Produce y_predictions that come from the X_validate
    y_pred = clf.predict(X_validate)
    
    class_report = classification_report(y_validate, y_pred,output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_validate, y_pred).ravel()
    
    fp_rate = fp/(fp+tn)
    fn_rate = fn/(fn+tp)
    tp_rate = tp/(tp+fn)
    tn_rate = tn/(fp+tn)
    # Compare actual y values (from validate) to predicted y_values from the model run on X_validate
    if print_results:
        print(f"-----VALIDATE/TEST RESULTS: {type(clf).__name__}-----")
        print(f"Using features: {X_validate.columns.to_list()}")
        print(f"Depth of {clf.max_depth}")
        print(f"Min Sample Leaf of {clf.min_samples_leaf}")
        print(classification_report(y_validate, y_pred))

        print(f'Accuracy on validate/test set: {accuracy:.2f}')
    validate_report = {'d':clf.max_depth, 
                       'm':clf.min_samples_leaf,
                    'accuracy':accuracy, 
                    'precision':class_report['1']['precision'], 
                    'recall':class_report['1']['recall'],
                   'fp_rate':fp_rate,
                   'fn_rate':fn_rate,
                   'tp_rate':tp_rate,
                   'tn_rate':tn_rate}
    
    return validate_report

def random_forest_train(X_train, y_train, selected_features, target, d = 10, m=1, print_results = True):
    """ Fits a Random Forest Classifier to train data and outputs the classification report and the classifier (clf) object.
    Takes in as arguments the split train data, a list of selected features, a string of the target name, a max_depth value (d), and min_sample_leaf value (m) """
  
    clf = RandomForestClassifier(max_depth=d, min_samples_leaf = m, random_state=123)
    clf = clf.fit(X_train, y_train)
    accuracy = clf.score(X_train, y_train)
    y_pred = clf.predict(X_train)
    class_report = classification_report(y_train, y_pred,output_dict=True)
    
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        
    fp_rate = fp/(fp+tn)
    fn_rate = fn/(fn+tp)
    tp_rate = tp/(tp+fn)
    tn_rate = tn/(fp+tn)
    if print_results:
        print(f"TRAINING RESULTS: {type(clf).__name__}")
        print(f"Using features: {X_train.columns.to_list()}")
        print(f"Depth of {clf.max_depth}")
        print(f"Min Sample Leaf of {clf.min_samples_leaf}")
        print("----------------")
        print(classification_report(y_train, y_pred))


        print(f"False positive rate: {fp/(fp+tn):.2%}")
        print(f"False negative rate: {fn/(fn+tp):.2%}")
        print(f"True positive rate: {tp/(tp+fn):.2%}")
        print(f"True negative rate: {tn/(fp+tn):.2%}")
        print("----------------")
    
    train_report = {'d':clf.max_depth, 
                    'm':clf.min_samples_leaf,
                    'accuracy':accuracy, 
                    'precision':class_report['1']['precision'], 
                    'recall':class_report['1']['recall'],
                   'fp_rate':fp_rate,
                   'fn_rate':fn_rate,
                   'tp_rate':tp_rate,
                   'tn_rate':tn_rate}
    
    return clf, train_report

def knn_train(X_train, y_train, selected_features, target, k=1, print_results = True):
    """Fits a K Nearest Neighbor Classifier to train data and outputs the classification report and the classifier (clf) object.
    Takes in as arguments the split train data, a list of selected features, a string of the target name, and a k value.
    """
    clf = KNeighborsClassifier(n_neighbors=k)
    clf = clf.fit(X_train, y_train)
    accuracy = clf.score(X_train, y_train)
    y_pred = clf.predict(X_train)
    class_report = classification_report(y_train, y_pred,output_dict=True)
    
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        
    fp_rate = fp/(fp+tn)
    fn_rate = fn/(fn+tp)
    tp_rate = tp/(tp+fn)
    tn_rate = tn/(fp+tn)
    if print_results:
        print(f"TRAINING RESULTS: {type(clf).__name__}")
        print(f"Using features: {X_train.columns.to_list()}")
        print(f"K of {clf.n_neighbors}")
        print("----------------")
        print(classification_report(y_train, y_pred))


        print(f"False positive rate: {fp/(fp+tn):.2%}")
        print(f"False negative rate: {fn/(fn+tp):.2%}")
        print(f"True positive rate: {tp/(tp+fn):.2%}")
        print(f"True negative rate: {tn/(fp+tn):.2%}")
        print("----------------")
    
    train_report = {'k':clf.n_neighbors, 
                    'accuracy':accuracy, 
                    'precision':class_report['1']['precision'], 
                    'recall':class_report['1']['recall'],
                   'fp_rate':fp_rate,
                   'fn_rate':fn_rate,
                   'tp_rate':tp_rate,
                   'tn_rate':tn_rate}
    
    return clf, train_report

def knn_validate(X_validate, y_validate, clf, print_results=True):
    """Evaluates k-Nearest Neighbors classifier models on validate (test) data. Takes as arguments the split validate data as well as the classifier object (clf) generated in the train function. Outputs the classification report with the results. """
    accuracy = clf.score(X_validate, y_validate)


    # Produce y_predictions that come from the X_validate
    y_pred = clf.predict(X_validate)
    
    class_report = classification_report(y_validate, y_pred,output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_validate, y_pred).ravel()
    
    fp_rate = fp/(fp+tn)
    fn_rate = fn/(fn+tp)
    tp_rate = tp/(tp+fn)
    tn_rate = tn/(fp+tn)
    # Compare actual y values (from validate) to predicted y_values from the model run on X_validate
    if print_results:
        # Later version should check if validate set or test
        print(f"-----VALIDATE/TEST RESULTS: {type(clf).__name__}-----")
        print(f"Using features: {X_validate.columns.to_list()}")
        print(f"K of {clf.n_neighbors}")
        print(classification_report(y_validate, y_pred))

        print(f'Accuracy on validate/test set: {accuracy:.2f}')
    validate_report = {'k':clf.n_neighbors, 
                    'accuracy':accuracy, 
                    'precision':class_report['1']['precision'], 
                    'recall':class_report['1']['recall'],
                   'fp_rate':fp_rate,
                   'fn_rate':fn_rate,
                   'tp_rate':tp_rate,
                   'tn_rate':tn_rate}
    
    return validate_report

def logistic_regression_train(X_train, y_train, selected_features, target, c=1, print_results = True):
    """Fits a Logistic Regression Classifier to train data and outputs the classification report and the classifier (clf) object. Takes in as arguments the split train data, a list of selected features, a string of the target name, and a C value.
    """
  
    clf = LogisticRegression(C=c)
    clf = clf.fit(X_train, y_train)
    accuracy = clf.score(X_train, y_train)
    y_pred = clf.predict(X_train)
    class_report = classification_report(y_train, y_pred,output_dict=True)
    
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        
    fp_rate = fp/(fp+tn)
    fn_rate = fn/(fn+tp)
    tp_rate = tp/(tp+fn)
    tn_rate = tn/(fp+tn)
    if print_results:
        print(f"TRAINING RESULTS: {type(clf).__name__}")
        print(f"Using features: {X_train.columns.to_list()}")
        print(f"C of {clf.C}")
        print("----------------")
        print(classification_report(y_train, y_pred))


        print(f"False positive rate: {fp/(fp+tn):.2%}")
        print(f"False negative rate: {fn/(fn+tp):.2%}")
        print(f"True positive rate: {tp/(tp+fn):.2%}")
        print(f"True negative rate: {tn/(fp+tn):.2%}")
        print("----------------")
    
    train_report = {'c':clf.C, 
                    'accuracy':accuracy, 
                    'precision':class_report['1']['precision'], 
                    'recall':class_report['1']['recall'],
                   'fp_rate':fp_rate,
                   'fn_rate':fn_rate,
                   'tp_rate':tp_rate,
                   'tn_rate':tn_rate}
    
    return clf, train_report

def logistic_regression_validate(X_validate, y_validate, clf, print_results=True):
    """ Evaluates logistic regression classifier models on validate (test) data. Takes as arguments the split validate data as well as the classifier object (clf) generated in the train function. Outputs the classification report with the results."""
    accuracy = clf.score(X_validate, y_validate)

    # Produce y_predictions that come from the X_validate
    y_pred = clf.predict(X_validate)
    
    class_report = classification_report(y_validate, y_pred,output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_validate, y_pred).ravel()
    
    fp_rate = fp/(fp+tn)
    fn_rate = fn/(fn+tp)
    tp_rate = tp/(tp+fn)
    tn_rate = tn/(fp+tn)
    # Compare actual y values (from validate) to predicted y_values from the model run on X_validate
    if print_results:
        # Later version should check if validate set or test
        print(f"-----VALIDATE/TEST RESULTS: {type(clf).__name__}-----")
        print(f"Using features: {X_validate.columns.to_list()}")
        print(f"C of {clf.C}")
        print(classification_report(y_validate, y_pred))

        print(f'Accuracy on validate/test set: {accuracy:.2f}')
    validate_report = {'c':clf.C, 
                    'accuracy':accuracy, 
                    'precision':class_report['1']['precision'], 
                    'recall':class_report['1']['recall'],
                   'fp_rate':fp_rate,
                   'fn_rate':fn_rate,
                   'tp_rate':tp_rate,
                   'tn_rate':tn_rate}
    
    return validate_report