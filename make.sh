#!/bin/sh

set -xe
# clang++ -Wall -Wextra -pedantic --std=gnu++2b pgrad.cc -o out/pgrad
clang++ -Wall -Wextra -pedantic *.cc -o out/autograd
rm -rf out/*.dSYM