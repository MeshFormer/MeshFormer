## Each column's meaning of features.

| Columns | Meaning                                   | Columns | Meaning                                 |
| :-----: | ----------------------------------------- | ------- | --------------------------------------- |
|   0-2   | `coordinates of each vertex(PCA)`       | 69-73   | `gauss curvature laplacian histogram` |
|   3-5   | `normals of each vertex(PCA)`           | 74      | `mean curvature`                      |
|  6-32  | `Fast point feature histogram`          | 75      | `maximum curvature`                   |
|  33-37  | `SDF histogram`                         | 76      | `minimum curvature`                   |
|  38-42  | `SIHKS`                                 | 77      | `gauss curvature`                     |
|   43   | `SDF VALUE`                             | 78-82   | `mean curvature histogram`            |
|  44-48  | `HKS`                                   | 83-87   | `maximum curvature histogram `        |
|  49-53  | `SDF laplacian histogram`               | 88-92   | `minimum curvature histogram`         |
|  54-58  | `mean curvature laplacian histogram`    | 93-98   | `gauss curvature histogram `          |
|  59-63  | `maximum curvature laplacian histogram` |      |
|  64-68  | `minimum curvature laplacian histogram` |         |                                         |

## Features calculation
* The average time of calculating features for each mesh (about 400k faces) is less than 1 mins. The code will be released soon. The histograms can be calculated fastly by the multiplication of sparse matrices.

## Visualization of features
* Run instruction like this.

```bash
python Analysis/Analysis_features.py \
    --input "${data_dir}/aliens"\
    --size 10 
```
