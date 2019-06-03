# Import CMake configuration for external projects.
# This repository uses pybind11 for a Python bindings header library
# and googletest for a C++ unit testing framework.
#
# Note that both of these directories are the result of a `git subtree` squash
# merge. (See https://www.atlassian.com/blog/git/alternatives-to-git-submodule-git-subtree)
# To update, do a `git subtree` pull. E.g.
#     git subtree pull --prefix pybind11 url master --squash
# Note that the scripted logic in this git tool can figure out the appropriate
# branch point for a new merge commit for references that are branches, but not
# for references that are tags. As such it is probably better to avoid using a
# git tag to reference a specific commit in the external project. Just use the
# master branch.

# Import the third-party Python bindings tools from the Pybind11 project: https://github.com/pybind
add_subdirectory(pybind11)

# Import the Googletest targets, unless the user has requested that the repository
# be downloaded instead (see tests/CMakeLists.txt)
if(NOT ${DOWNLOAD_GOOGLETEST})
    if(${BUILD_TESTING})
        add_subdirectory(googletest EXCLUDE_FROM_ALL)
    endif()
endif()
