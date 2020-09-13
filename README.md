## Architecture

![Architecture](images/cwicarc.png)


## Training your own model

To train the model, you need to supply it with a dataset of high quality RGB images and a set of validation images to track the reconstruction.
The data is read through a glob pattern which must expand to a list of RGB images in PNG format.
Training can be as simple as the following command:

````bash
python train.py --verbose train --train_glob="train_images/*.png" --valid_glob="valid_images/*.png"
````

