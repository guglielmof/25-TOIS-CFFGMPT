import pandas as pd


class AbstractPredictor:

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def predict(self, queries):
        if hasattr(self, 'hyperparams_identifiers'):
            params_string = "_".join([str(self.__getattribute__(name)) for name in self.hyperparams_identifiers])
        else:
            params_string = None
        predictions_values = queries.apply(self._local_predict, axis=1)
        predictions = pd.DataFrame({"query_id": queries.query_id.to_list(), "prediction": predictions_values})
        predictions["predictor"] = self.predictor_name
        predictions["params"] = params_string
        print(predictions)
        return predictions

    def _local_predict(self, query):
        raise NotImplementedError
