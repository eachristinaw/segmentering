# Segmentering

Segmenteringsmodellen er udviklet i Python. Modellen kan hentes direkte fra dette repository. Herunder er et Python eksempel, hvor modellen anvendt på et mindre datasæt.

Jeg er for sej. 

## Spørgsmål
Segmenteringsmodellen er baseret på disse ti spørgsmål:

1. Vi burde gøre mere for flygtninge, der kommer til Danmark, end vi gør i dag
2. Den offentlige sektor er for stor
3. Vi betaler for meget i skat i Danmark
4. Vi skal have styr på vores egne problemer, før vi hjælper andre lande
5. Enhver er sin egen lykkes smed
6. Jeg køber altid økologiske eller miljøvenlige produkter, hvis jeg kan
7. Det er vigtigt for mig at have den nyeste teknologi på markedet
8. Jeg kan ikke forestille mig en hverdag uden min smartphone
9. Jeg kommer let til at kede mig, hvis jeg laver de samme ting
10. Jeg vil følge moden

I data skal disse navngives sådan at spørgsmål 1 hedder Q1, spørgsmål 2 hedder Q2 osv.

Svarmulighederne på spørgsmålene er:
1. Fuldstændig uenig
2. Uenig
3. Nærmest uenig
4. Nærmest enig
5. Enig
6. Fuldstændig enig


## Kode
Importerer de nødvendige Python biblioteker:
```python
import pickle
import pandas as pd
from sklearn.cluster import KMeans
```
### Data
Indlæser og inspicerer data:
```python
data = pd.read_csv("data.csv")
data.head()
```
|                Q1 |               Q2|...|            Q9|               Q10|
|-------------------|-----------------|---|--------------|------------------|
| Fuldstændig uenig | Fuldstændig enig|...|Nærmest uenig |             Uenig|
|     Nærmest uenig |             Enig|...|Nærmest uenig |     Nærmest uenig|
|              Enig |    Nærmest uenig|...|        Uenig |             Uenig|
|      Nærmest enig |    Nærmest uenig|...|Nærmest uenig |             Uenig|
|      Nærmest enig |            Uenig|...|        Uenig | Fuldstændig uenig|

I dette datasæt er svarene i tekstformat. Kmeans-modellen kan ikke anvendes på tekstdata. Derfor skal teksten i kolonnerne ændres til numeriske værdier:

```python
replace_dict = {"Fuldstændig uenig": 1,
                "Uenig": 2,
                "Nærmest uenig": 3,
                "Nærmest enig": 4,
                "Enig": 5,
                "Fuldstændig enig": 6}

data.iloc[:,:] = data.iloc[:,:].replace(replace_dict)
data.head()
```
| Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 |
|----|----|----|----|----|----|----|----|----|-----|
| 1  | 6  | 2  | 6  | 6  | 4  | 3  | 4  | 3  | 2   |
| 3  | 5  | 6  | 5  | 5  | 3  | 3  | 6  | 3  | 3   |
| 5  | 3  | 5  | 4  | 5  | 1  | 2  | 4  | 2  | 2   |
| 4  | 3  | 3  | 3  | 4  | 1  | 2  | 2  | 3  | 2   |
| 4  | 2  | 2  | 3  | 3  | 3  | 3  | 4  | 2  | 1   |

Data er nu numerisk.

### Model
Modellen loades og benyttes på data:
```python
with open("kmeans.pickle", "rb") as f:
    kmeans = pickle.load(f)

data["segment"] = kmeans.predict(data)
```

Segmenterne kommer i første omgang ud som numeriske værdier. Her ændres værdierne til navnene på segmenterne:
```python
replace_dict_segment = {0: "Selvudviklerne",
                        1: "Individualisterne",
                        2: "Pragmatikerne",
                        3: "Idealisterne",
                        4: "Beskytterne"}

data["segment"] = data["segment"].replace(replace_dict_segment)
data.head()
```

| Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 | segment           |
|----|----|----|----|----|----|----|----|----|-----|-------------------|
| 1  | 6  | 2  | 6  | 6  | 4  | 3  | 4  | 3  | 2   | Beskytterne       |
| 3  | 5  | 6  | 5  | 5  | 3  | 3  | 6  | 3  | 3   | Individualisterne |
| 5  | 3  | 5  | 4  | 5  | 1  | 2  | 4  | 2  | 2   | Pragmatikerne     |
| 4  | 3  | 3  | 3  | 4  | 1  | 2  | 2  | 3  | 2   | Pragmatikerne     |
| 4  | 2  | 2  | 3  | 3  | 3  | 3  | 4  | 2  | 1   | Idealisterne      |
