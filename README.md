# make-it-move

## Setting up

You can create an appropriate `conda` environment with the following command:

```
conda env create -f environment.yml
```

You should then be able to activate this environment with:

```
conda activate make-it-move
```
## Data Collection

We use `selenium` to do data retrieval from `website`. You can run 
```
python data_collection.py
```
to update the meme dataset. Data is stored in the `Data` folder.