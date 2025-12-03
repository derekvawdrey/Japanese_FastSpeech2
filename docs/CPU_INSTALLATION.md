1. Create a conda environment and run the following, this allows you to preprocess
```
conda create -n fastspeech python=3.10
conda activate fastspeech
conda install -c conda-forge numpy=1.26.4 scipy=1.13.1 numba=0.59
conda install -c pytorch pytorch torchvision torchaudio cpuonly
conda install -c conda-forge \
  absl-py audioread cachetools certifi cffi charset-normalizer click contourpy \
  cycler dataclassy decorator Distance filelock fonttools fsspec g2p-en \
  google-auth google-auth-oauthlib grpcio idna importlib_metadata importlib_resources \
  inflect jinja2 joblib kiwisolver kneed librosa llvmlite markdown markdown-it-py \
  markupsafe matplotlib mdurl montreal-forced-aligner mpmath networkx nltk \
  oauthlib packaging pandas pillow praatio protobuf pyasn1 pyasn1-modules pycparser \
  pygments pyparsing pypinyin python-dateutil pytz regex requests \
  requests-oauthlib resampy rich rich-click rsa scikit-learn seaborn six \
   SQLAlchemy sympy tensorboard tensorboard-plugin-wit threadpoolctl \
  tqdm typing_extensions tzdata Unidecode werkzeug zipp

pip install tgt

conda install -c conda-forge pyworld
```