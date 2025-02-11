import faiss


class FaissIndex:

    def __init__(self, model_name=None, base_path=None, corpus=None, path=None, data=None, mapper=None):
        self.index_type = "faiss"

        if path is not None:
            self._load_given_path(path)
        elif not (base_path is None or corpus is None or model_name is None):
            path = f"{base_path}/faiss/{corpus}/{model_name}/{model_name}"
            self._load_given_path(path)
        elif not (data is None or mapper is None):
            self._construct_from_data(data, mapper)


        else:
            raise ValueError("when constructing a FaissIndex, either specify the parameter path, or all the parameters base_path, corpus, model_name")




    def _load_given_path(self, path):

        self.index = faiss.read_index(f"{path}.faiss")

        self.mapper = list(map(lambda x: x.strip(), open(f"{path}.map", "r").readlines()))


    def _construct_from_data(self, data, mapper):
        self.index = faiss.IndexFlatIP(data.shape[1])
        self.index.add(data)
        self.mapper = mapper
