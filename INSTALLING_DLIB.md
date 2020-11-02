# Installing DLIB

## Things to know before starting
There a few things to know before starting:
 * Compiling with CUDA support requires a valid CUDA SDK installed, with a version that is supported by DLIB. You can find the changes for each CUDA version [here](http://dlib.net/release_notes.html).
 * Compiling to use native AVX instructions on your CPU requires a flag to be parsed, specifically the `-D DLIB_USE_AVX_INSTRUCTIONS=1` you can change the `1` to a `0` to disable AVX instruction support in the source compilation down below.
 * **Once you compile with CUDA support, you cannot disable the use of your graphics card if DLIB decides to use it. You will only be able to run 1 thread effectively unless you compile from source without CUDA support or install from pip (Which usually doesn't install CUDA support)**

## Steps before installation
### Windows
 * Install Visual Studio 2015 C++ ([Download here](https://www.microsoft.com/en-au/download/details.aspx?id=48145))
 * Install cmake for windows ([Download here](https://cmake.org/download/))

### Linux
 * Install g++ (Or any C++11 compiler)
 * Install cmake

## MacOS
 * Install the `brew` package manager from [here](https://brew.sh) and run the following commands:

```bash
brew cask install xquartz
brew install gtk+3 boost
brew install boost-python --with-python3
brew install dlib
```

# Installing from source
Follow these commands after setting up for your relevant operating system.
```bash
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build && cd build
```

*To compile **with** CUDA support, run this command*

```bash
cmake -D DLIB_USE_CUDA=1 -D USE_AVX_INSTRUCTIONS=1 ..
```


*To compile **without** CUDA support, run this command*

```bash
cmake -D DLIB_USE_CUDA=0 -d USE_AVX_INSTRUCTIONS=1 ..
```

```bash
cmake --build . --config Release`
cd .. && python setup.py install
```

# Installing from pip
Follow this command after setting up for your relevant operating system.
```bash
pip install --upgrade dlib
```

If you get a permissions error, run the command like this:
```bash
pip uninstall dlib
pip install --upgrade dlib --user
```