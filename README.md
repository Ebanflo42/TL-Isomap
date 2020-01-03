# TL-Isomap

A Python library implementing the Topological Landmark-Isomap algorithm presented in "Homology-Preserving Dimensionality Reduction via Manifold Landmarking and Tearing" by Yan et al.

## Getting Started

Clone this repo and move the base director wherever you wish. This library is not yet installable via `pip` or `conda` so you must import it like so:

```
import sys

sys.path.append('/path/to/TL-Isomap/src')

import tl_isomap
```

## Further Information

The library offers the use of DBSCAN or SLC for clustering; different parameters must be passed to the constructor of the `Mapper` object depending on which clustering algorithm is to be used.

In the future, manifold tearing ought to be added to the library.
