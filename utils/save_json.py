import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_analytics_json(path, analytics):
    with open(path, 'w') as f:
        f.write(json.dumps(analytics, cls = NumpyEncoder))

def save_annotations_json(path, annotations):
    with open(path, 'w') as f1:
        f1.write(json.dumps(annotations, cls = NumpyEncoder))

