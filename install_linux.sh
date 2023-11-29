pip install -r requirements_pt1.txt
cd qwrapper
pip install -r requirements.txt
pip install .
cd ../qml
pip install -r requirements.txt
pip install .
cd ../benchmark
pip install -r requirements.txt
pip uninstall jax
pip uninstall jaxlib
pip install .
cd ../qswift
pip install -r requirements.txt
pip install .
cd ..
pip install -r requirements_pt2.txt
pip install autograd
