# trajectory_prediction_team_40

Argoverse 2 251B kaggle competion for trajectory prediction

# Install dependencies

To create the env

```
conda env create -f environment.yml
conda activate tp
```

To update the env and pull new deps

```
conda env update -f environment.yml
```

Environement check

```
conda activate tp

python smoke.py
```

For kaggle api - able to download / submit stuff

Rename `kaggle.json.template` to `kaggle.json`
Update your info

```
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_KAGGLE_KEY"
}

```

`cp kaggle.json ~/.kaggle/kaggle.json`

`chmod 600 ~/.kaggle/kaggle.json`

`rm kaggle.json`

What you can do with it

```
kaggle competitions files -c cse-251-b-2025 //test

kaggle competitions download -c cse-251-b-2025 -p data/ --unzip

kaggle competitions submit -c cse-251-b-2025 \
    -f predictions.csv \
    -m "my model v1"

```

To view RESEARCH.md flow chart install mermaid preview https://marketplace.visualstudio.com/items/?itemName=bierner.markdown-mermaid. Then `cmd-shift-v`.
