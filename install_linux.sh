pip install -r requirements_pt1.txt
cd qwrapper
pip install -r requirements.txt
pip install .
cd ../qswift
pip install -r requirements.txt
pip install .
cd ..
pip install -r requirements_pt2.txt
pip uninstall jax
pip uninstall jaxlib
pip install autograd
pip install numpy==1.23.0
pip install cupy-cuda12x
pip install cuda-quantum
