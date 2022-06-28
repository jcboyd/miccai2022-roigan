## Configuration options

### Experiment options

* ```hes_dir``` : Specifies the path to the directory containing HES data.
* ```ihc_dir``` : Specifies the path to the directory containing IHC data.
* ```hes_library``` : Specifies the path to the directory containing HES library data.
* ```ihc_library``` : Specifies the path to the directory containing IHC library data.

### Training options

* ```nb_epochs``` : Number of training epochs.
* ```steps_per_epoch``` : Number of batches per epoch.
* ```nb_batch``` : Number of training samples (tiles) per batch.
* ```nb_rois``` : Number of rois per batch (divided evenly among batch tiles).

### Model options

* ```nb_residuals``` : Number of residual blocks per generator.
* ```lambda_cyc``` : Hyperparameter weight for cycle consistency (L1) loss.
* ```lambda_id``` : Hyperparameter weight for identity (L1) loss.
* ```lambda_roi``` : Hyperparameter weight for ROI GAN loss.