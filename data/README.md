# Data

## Questionnaire (included)

`dass21_responses.csv` contains 580 anonymous DASS-21 questionnaire responses: one column per statement, each answered on a 0-3 scale. There are no names or other personal details. Notebook 1 uses the seven anxiety items.

## Face images (not included)

The face datasets are distributed under their own licence terms, so they are not redistributed here. Download them and arrange them like this:

```
data/images/
├── ck+48/
│   ├── anger/  contempt/  disgust/  fear/  happy/  sadness/  surprise/
└── kdef/
    ├── angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
```

- **CK+48**: a 48x48 grayscale version of the Extended Cohn-Kanade (CK+) dataset, available on Kaggle as [CKPLUS](https://www.kaggle.com/datasets/shawon10/ckplus). It already comes with the folder layout above.
- **KDEF**: the Karolinska Directed Emotional Faces, available after registration at [kdef.se](https://kdef.se/). Sort the extracted images into emotion folders with:

  ```bash
  python data/organize_kdef.py path/to/KDEF
  ```

  By default only front-facing pictures are copied; add `--all-angles` to include the profile views. Notebook 2 converts every image to 48x48 grayscale when loading it.
