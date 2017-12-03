#!/bin/bash

script_path=`dirname "$0"`
project_path=`dirname "$script_path"`
sources=""
for src in $(find "$project_path"  -name *.c -o -name *.h | sort -f); do
    dir=`dirname "$src"`
    dir=`basename "$dir"`
    if ! [ "$dir" = "tmp" ]; then
        sources="$sources $src"
    fi
done
#for src in $(find "$project_path" -name *.h); do
#    dir=`dirname "$src"`
#    dir=`basename "$dir"`
#    if ! [ "$dir" = "tmp" ]; then
#        sources="$sources $src"
#    fi
#done
if [ "$1" = "--with-makefiles" ]; then
    for makefile in $(find "$project_path" -name Makefile); do
        sources="$sources $makefile"
    done
    for makefile in $(find "$project_path" -name *.mk); do
        sources="$sources $makefile"
    done
fi
cmd=""
os=`uname`
if [ "$os" = "Linux" ]; then
    cmd="xdg-open"
elif [ "$os" = "Darwin" ]; then
    cmd='open' 
fi
if [ -z "$cmd" ]; then
    cmd=vim
fi
`$cmd $sources`
