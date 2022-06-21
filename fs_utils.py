import random 
from numpy import array
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def n_to_f(n, nfeats):
    res = [int(i) for i in list('{0:0b}'.format(n))]

    to_add = nfeats - len(res)
    r = [0 for _ in range(to_add)]

    res = r+res
    # printing result 
    return res


def get_cols(df, fo):
    dfs = df.iloc[:, 1:-1]
    colies = [dfs.columns[x] for x in range(len(fo)) if fo[x]]
    #print(colies)
    return df[[df.columns[0]]+colies+[df.columns[-2]]+[df.columns[-1]]]
    
    
def eval_f(df, fo):
    if sum(fo) == 0:
        return 0 
    dfs = get_cols(df, fo)
    X = (array(dfs)[:,1:-1].astype(float))
    y = array(dfs.y)
    clf = RandomForestClassifier()
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
    return scores.mean()

def random_move(of):
    fo = of[:]
    i = random.randint(0,len(fo)-1)
    fo[i] ^= 1 
    return fo
    