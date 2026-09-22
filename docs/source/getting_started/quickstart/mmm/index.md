# MMM Quickstart

```python
import pandas as pd

from pymc_marketing.mmm import (
    GeometricAdstock,
    LogisticSaturation,
    MMM,
)
from pymc_marketing.paths import data_dir

file_path = data_dir / "mmm_example.csv"
data = pd.read_csv(file_path, parse_dates=["date_week"])

mmm = MMM(
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    date_column="date_week",
    channel_columns=["x1", "x2"],
    control_columns=[
        "event_1",
        "event_2",
        "t",
    ],
    yearly_seasonality=2,
)
```

Once the model is fitted, we can further optimize our budget allocation as we are including diminishing returns and carry-over effects in our model.

New to MMM? Follow the {ref}`guided learning path <mmm_learning_path>` first. It tells you what to read, what to run, and in which order.

Explore a hands-on {ref}`simulated example <mmm_example>` for more insights into MMM with PyMC-Marketing.
