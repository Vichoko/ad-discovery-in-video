# ad-discovery-in-videos
Finding advertising clips in a full-length video.

# Dependencies
* Python 3.6
* Numpy
* OpenCV

# Usage
```
python adlookup.py <tv-video-filename> <ad-foldername>
```

TV Video & Ad Folder should be in ```./data/``` or in path set in DATA_FOLDER (Refer to [Config](#configuration file)

## Example
```
python adlookup.py mega-2014_04_23.mp4 ads
```

# Configuration file

in ```./src/configurations.py```
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

