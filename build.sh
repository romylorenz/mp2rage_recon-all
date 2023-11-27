#!/bin/bash
shopt -s extglob

export SINGULARITY_TMPDIR=/tmp
export SINGULARITY_CACHEDIR=/tmp

deffile=${1-gfae.def}
container_fname=$(basename $deffile .def).sif


# BUILD CONTAINER
tmpdir=$(mktemp -d)
cwd=$(pwd)

cp -r !(*.sif) $tmpdir/

cd $tmpdir

singularity build --fakeroot $container_fname $deffile
mv $container_fname $cwd/
cd $cwd

rm -rf $tmpdir
