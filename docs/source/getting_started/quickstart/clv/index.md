# CLV Quickstart

We can choose from a variety of models, depending on the type of data and business nature. Let us look into a simple example with the Beta-Geo/NBD model for non-contractual continuous data.

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymc_marketing import clv

data_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/clv_quickstart.csv"
data = pd.read_csv(data_url)
data["customer_id"] = data.index

beta_geo_model = clv.BetaGeoModel(data=data)

beta_geo_model.fit()
```

Once fitted, we can use the model to predict the number of future purchases for known customers, the probability that they are still alive, and get various visualizations plotted.

See the {ref}`gallery` section for more on this.
