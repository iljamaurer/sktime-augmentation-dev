# Augmenter Development Repo
### Load Subrepos
```
git submodule update --init --recursive
cd sktime_dev
git checkout seq-augmentation
```

## Docker
### Build
```
docker build sktime/ -f sktime/.binder/Dockerfile -t sktime-base
```

```
docker build . -t sktime-dev
```

### Run
Linux
```
docker run -ti -v ${PWD}:/code sktime-dev /bin/bash
```
Windows
```
docker run -ti -v %cd%:/code sktime-dev /bin/bash
```


## Unit Tests
https://www.sktime.org/en/v0.4.2/contributing.html#unit-testing

```
cd /code/sktime_dev
```

all tests:
```
pytest sktime/
```

specific tests, e.g.
```
pytest sktime/transformations/
```


## Contribute
https://www.sktime.org/en/latest/get_involved/contributing.html#contributing

