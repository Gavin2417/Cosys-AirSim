cd utils/nearest_neighbors
python setup.py build_ext --inplace
cd ../../

cd utils/cpp_wrappers
sh compile_wrappers.sh
cd ../../../