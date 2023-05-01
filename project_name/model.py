from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier

# This is where we would implement our custom model

def get_model(args):
    if args.model_name == 'decision_tree':
        model = DecisionTreeClassifier(max_depth=args.max_depth)
    elif args.model_name == 'ridge':
        model = RidgeClassifier(alpha=args.alpha)
    else:
        raise ValueError('Invalid model_name: {}'.format(args.model_name))
    return model