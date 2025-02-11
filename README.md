# Create the indexes

Use the script <code>MYRETRIEVE.code.indexing_simple</code> to create the index. There are 2 parameters that are 
mandatory, the collection to be considered and the encoder, set respectively with <code>-c</code> and <code>-e</code>.
The encoders are those available in the package <code>MYRETRIEVE.code.irmodels</code>. The name of the collection
must correspond to a valid id of <a href=https://ir-datasets.com/>ir_datasets</a>. There are also three optional
parameters, the name of the corpus (<code>--corpus_name</code>, default: same as collection)
the batch size (<code>-b</code>, default: 5000) and the path where
the index will be saved (<code>--collections_path</code>, default: collections).

For example, assume you want to create  the index for the deep learning 2019 passage collection.

The call is:
```bash 
python code/MYRETRIEVE/code/indexing_simple.py -c msmarco-document/trec-dl-2019/judged --corpus_name msmarco -e contriever
```
this will create a memmap of the index of the MSMARCO passage collection in 
<code>collections/INDEXES/memmap/msmarco/contriever</code>.

Repeat this process for all the encoders and collections you want to do the predictions for.
Importantly, the script relies on ir_datasets and thus it follows the same rules (for example, if you want to index the
tipster corpus, it must be available already and it is not downloaded).

Notice that, for full replicability of the results reported in the paper, it would be necessary also to compute 
pyterrier indexes of the corpora (putting them in <code>collections/INDEXES/pyterrier</code>). We do not provide the 
code here and the interested practitioner can follow the pyterrier tutorial available here 
<a href="https://pyterrier.readthedocs.io/en/latest/terrier-indexing.html">
https://pyterrier.readthedocs.io/en/latest/terrier-indexing.html</a>.


# Set up the properties.ini file
Within <code>properties.ini</code> there are the properties/hyperparameters that need to be set to ensure replicability of the experiments.
the most important value is <code>indexing_dir</code> that must be equal to the part of the path before "memmap" in the 
indexing path (see above).

# Create the runs

Once the indexes are available, it is necessary to precompute the runs. As for the index, specify the collection and the 
encoder. For the encoders, the rules remain the same as to create the index. For what concerns the collection, 
on the other hand, we use slightly different policy. In fact, valid collection names are those in the <code>
[Collections]</code> filed of the properties.ini file. The properties.ini file defines the mapping between the collection
name and the corresponding ir_datasets id and the function to read and process the queries and the qrels.
Use the following command to compute a run:
```bash 
python code/preliminary/retrieve.py -c trec-dl-2019 -e contriever
```

The run will be stored in <code>data/runs</code>. Notice that, for the sake of replicability, we uploaded in this
repository a set of precomputed runs corresponding to those used in the experimental part of the paper.

The WRIG predictor requires to compute other runs, using the query reformulations constructed using word2vec. Therefore,
following the same procedure used for the retrieve.py script, to compute the WRIG predictions it is necessary to also run
the retrieve_variations,py script, to compute the run for the query variations. 


# Compute the predictions

The script that allows to compute the prediction is <code>predict.py</code>  that takes in input four parameters: the
considered predictor (<code>-p</code>), the collection (<code>-c</code>), the encoder (<code>-e</code>) and the number of parallel processes used to 
compute the predictions (<code>-w</code>). The name of the predictor must correspond to one of the classes in the qpp 
package. The proposed predictor is PDQPP. As for the runs, the collection must be one of the collection IDs in the 
properties.ini file. The same rules as for  the runs and indexes apply to the encoders. The number of parallel processes 
is set by default to 40. 

To compute the PDQPP predictions for the trec-dl-2019 collection and the contriever encoder, the command is:
```bash 
python code/experiments/predict.py -p PDQPP -c trec-dl-2019 -e contriever
```

# Validate the hyperparameters and print the results

When the predictions for all QPPs have been computed, it is necessary to select the optimal hyperparameters and compare 
the predictors. to do so, use the script <code>print_perf_table.py</code>. While the print_perf_table.py script allows 
the practitioner to set a number of parameters (the collections considered, the IR measure, the encoders/ir models and the qpp 
measures), the default values of such parameters correspond to the ones used in the paper. 