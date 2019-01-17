#!/usr/bin/env bash
# Install a newer CMake to meet GROMACS minimum cmake version requirement on
# Travis-CI Ubuntu 14 container.
# Reference https://riptutorial.com/cmake/example/4723/configure-travis-ci-with-newest-cmake

set -ev

mkdir -p ${CMAKE_ROOT}

CMAKE_BUILD=${TRAVIS_BUILD_DIR}/cmake-dist
mkdir -p ${CMAKE_BUILD}

pushd ${CMAKE_BUILD}
# we use wget to fetch the cmake binaries
wget --no-check-certificate https://github.com/Kitware/CMake/releases/download/v3.13.3/cmake-3.13.3.tar.gz
# this is optional, but useful:
# do a quick checksum to ensure that the archive we downloaded did not get compromised
echo "665f905036b1f731a2a16f83fb298b1fb9d0f98c382625d023097151ad016b25  cmake-3.13.3.tar.gz" > cmake_sha.txt
sha256sum -c cmake_sha.txt
# extract the binaries; the output here is quite lengthy,
# so we swallow it to not clutter up the travis console
tar -xvf cmake-3.13.3.tar.gz > /dev/null
mv cmake-3.13.3 cmake-install
# don't forget to switch back to the main build directory once you are done
popd
