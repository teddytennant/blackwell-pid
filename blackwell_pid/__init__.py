import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from blackwell_pid.closed_form import (
    channel_posterior,
    mutual_info,
    normalized_posterior,
    redundancy_fastpath,
    union_closed_form,
)

__all__ = [
    "mutual_info",
    "channel_posterior",
    "normalized_posterior",
    "union_closed_form",
    "redundancy_fastpath",
]
