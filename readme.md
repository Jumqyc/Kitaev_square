Basic feature:
We use pybind11 to expose a C++ class to python. Now you can directly run the monte carlo simulation in python, but its implementation is in C++! 

How to use it? 
Well the module is called ising, and the class is called Ising. First import the python module:
>> import ising
Now, you can construct the object like an ordinary python class:
>> model = ising.Ising(10, 10)
this will create a object, with size 10x10, now you can use 
>> model.set_parameters(temperature = 3.0,J1 = 1.0,J2 = 1.0,J3 = 0.2)
to set the parameters. The initial spin are all spin up. 

How to run the simulations? 
Simple! 
>> model.run(Nsample = 10, spacing = 10)
this will run 100 monte carlo sweeps (that is, Lx*Ly*100 local updates), and take 10 sample, with spacing between each sample as 10.

How to get the data? 
>> model.get_energy()       # gives the recorded energy
>> model.get_magnetism()    # gives the recorded magnetism
>> model.get_afm()          # gives the recorded afm order
this will output an numpy array of length Ntest, with Ntest = the number of samples you have taken. 

How to record the data? 
Ising class is compatible with pickle. You can dump it into some pkl file. 

How to create a python module using C++?
Step 0: Download CMake.
Step 1: copy pybind11 folder under extern folder. You can find the latest release in github
Step 2: Create an empty folder "build"
Step 3: Copy CMakeLists.txt
Step 4: run the following code in cmd: 
For Windows,
>> cd build
>> cmake .. 
>> cmake --build . --config Release
For Mac/Linux,
>> cd build
>> cmake .. -DCMAKE_BUILD_TYPE=Release
>> make
Step 5: Now the released file should in ./build/Release

The file wrapper.cpp will wrap Ising.cpp to a python module. Ising.cpp records how the file runs in C++, while wrapper.cpp record how it is exposed to python. 