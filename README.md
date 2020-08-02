# pymlkits

* Author: congvm
* Email: congvm.it@gmail.com

---

### Introduction

Common tools in Python

* images: functions to work on images
* multiprocessing: parallel processes
* systems: system handlers
* visualization: tools to visualize images

### Distribution

```
pip install twine
python setup.py sdist bdist_wheel
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*  --verbose
```