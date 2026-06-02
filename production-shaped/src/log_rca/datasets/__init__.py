"""Per-dataset loaders. Each loader keeps its source's natural schema.

We deliberately do **not** unify schemas across datasets: real-world
sources have different columns (Airflow has ``dag_id``/``task_id``;
Hadoop has ``Process``/``Component``; HDFS has ``Pid``/``Component``),
and the pipeline runs against one dataset at a time so we can compare
results — not against a smeared union.

Adding a new dataset:
1. Create a new module ``loghub_<name>.py`` (or ``my_source.py``).
2. Define a frozen dataclass for its record type.
3. Expose a ``...Loader`` class with ``load_records()`` and ``summarise()``.
4. Document it in ``data/README.md``.
"""

from log_rca.datasets.loghub_hadoop import (
    HadoopRecord,
    LogHubHadoopLoader,
)
from log_rca.datasets.synthetic_airflow import (
    SyntheticAirflowLoader,
    SyntheticAirflowRecord,
)

__all__ = [
    "SyntheticAirflowLoader",
    "SyntheticAirflowRecord",
    "LogHubHadoopLoader",
    "HadoopRecord",
]
