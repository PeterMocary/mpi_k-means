# K-means using OpenMPI

The k-means algorithm implementation which clusters regular
integers (0-255) taken from a file `numbers` byte by byte.
The implementation uses OpenMPI to paralellize the algorithm.

## How to use

- Requires OpenMPI installed.
- Use ``make`` to build the program.
- To run use ``bash ./run.sh <numbers_cnt>`` where ``<numbers_cnt>`` is a
  number from range 4 - 32. These numbers will be randomly generated and 
  the input file will be removed after the program runs.

```
make
bash ./run.sh 32
```

- To run the ``test.py`` script, ``scikit-learn`` is required.
- ``python ./test.py <numbers_cnt> <tests_cnt>`` where ``<numbers_cnt>`` is a number from
  range 4-32 and ``<tests_cnt>`` is number of tests to run.

```
python3 -m venv .venv
source .venv/bin/activate
pip install scikit-learn
python ./test.py 32 10
```
