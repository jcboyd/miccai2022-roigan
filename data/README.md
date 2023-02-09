## Library building

The `library_demo.ipynb` notebook illustrates how to extract a weak ground truth of healthy and cancer cell bounding boxes from H&E patches. The procedure is essentially identical for IHC images, except the expert annotation is replaced with an automatically derived mask from the DAB channel. This code can be wrapped in a set of for-loops for batch processing.
