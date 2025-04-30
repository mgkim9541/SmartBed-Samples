from Codes.preprocessing import preprocess
import pandas as pd

def test_preprocess():
    df = pd.DataFrame({'Com':[0,1], 'A':[1,2]})
    X_train, X_test, y_train, y_test = preprocess(df, target='Com')
    assert X_train.shape[0] == 1
