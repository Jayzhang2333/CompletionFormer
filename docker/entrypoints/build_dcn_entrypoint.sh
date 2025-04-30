#!/bin/bash
# Only build if not already built
if [ ! -f /completion_former/src/model/deformconv/build/lib.linux-x86_64-3.8/deform_conv.cpython-38-x86_64-linux-gnu.so ]; then
    cd /completion_former/src/model/deformconv
    python setup.py build install
fi

exec "$@"
