# TL-Isomap

A Python library implementing the Topological Landmark-Isomap algorithm presented in "Homology-Preserving Dimensionality Reduction via Manifold Landmarking and Tearing" by Yan et al.

## Getting Started

Clone this repo and move the base director wherever you wish. This library is not yet installable via `pip` or `conda` so you must import it like so:

```
import sys

sys.path.append('/path/to/TL-Isomap/src')

import tl_isomap
```

## Example

```
$ cd TL-Isomap
$ python
>>> import sys
>>> sys.path.append('src')
>>> import numpy as np
>>> import clustering as cl
>>> import filter_functions as ff
>>> import tl_isomap as tli
>>> data = np.load('test_data/spirals.npy')
>>> my_tli = tli.TLIsomap(data, ff.eccentricity_p(data, 2.0), cl.DBSCAN, num_bins = 10, dbscan_eps = 0.5, num_neighbors = 4, isomap_eps = 0.1)
>>> my_tli.run_mapper()
>>> my_tli.graph.edges
 . . .
>>> embedding = my_tli.embed_k_dims(2) #an expensive computation
>>> embedding
 . . .
```

## Further Information

The library offers the use of DBSCAN or SLC for clustering; different parameters must be passed to the constructor of the `TLIsomap` object depending on which clustering algorithm is to be used.
