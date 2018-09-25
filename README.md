# ad-discovery-in-videos
Finding advertising clips in a full-length video.

# Dependencies
* Python 3.6
* Numpy
* OpenCV

## Other requirements
Need at least one TV Video and some AD Videos to be looked up within the
TV Video, inside the ```DATA_FOLDER```.

The structure of ```DATA_FOLDER``` should be, for instance:
```
data/
-- mega-2014_04_11.mp4
-- ads/
----- ballerina.mpg
----- cristal.mpg
```

By default, data folder is included with this project; so you can add
your videos to ```./data/```, or modify the path set by ```DATA_FOLDER``` variable
(Refer to Configuration file)

# Usage
```
python adlookup.py <tv-video-filename> <ad-foldername>
```

TV Video & Ad Folder should be in ```./data/```, or in path set in
```DATA_FOLDER``` (Refer to Configuration file).

## Example usage
```
python adlookup.py mega-2014_04_11.mp4 ads
```

Wait until it finishes.
## Evaluation
Using TV and AD videos from [Google Drive](https://drive.google.com/drive/folders/1suHYlStIt0Bj4D3pmncANcZymzcE6bwm),
you can evaluate the performance of the solution with:

```
python evaluar.py detecciones.txt
```

Read the metrics in STDOUT.


# Configuration file

In ```./src/configurations.py```

Can be changed if needed.

```
# GENERAL
DATA_FOLDER = Path("data/")
CACHE_FOLDER = Path("cache/")
SUPPORTED_EXTENSIONS = ["mpg", "mp4"]

# FEATURE EXTRACTION
SAMPLES_PER_SECOND = 2
SOBEL_THRESH = 100
SAMPLING_DIMENSIONS = (32, 32)

# KNN
K = 5

# AD DETECTION
APPEARANCES_OUTFILE = "detecciones.txt"
SCORE_THRESHOLD = 0.25
```

