from ..AbstractPredictor import AbstractPredictor
import pyterrier as pt
from .IndexWrapper import IndexWrapper


class AbstractPTPredictor(AbstractPredictor):

    def __init__(self, *args, **kwargs):

        # the parameters, such as the run and the index and the hyperparameters are in kwargs, AbstractPredictor creates the fields programmatically
        super().__init__(self, *args, **kwargs)


        self.index_wrapper = IndexWrapper(kwargs["index"], stemmer=pt.TerrierStemmer.porter, stopwords=kwargs["stoplist"])
