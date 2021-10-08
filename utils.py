import pickle


def save_pkl(path, obj):
    with open(path, 'wb') as writer:
        pickle.dump(obj, writer, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as reader:
        obj = pickle.load(reader)
    return obj
