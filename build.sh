#!/bin/bash
shopt -s extglob

build_location=$1
deffile=mp2rage_recon-all.def
container_fname=$(basename $deffile .def).sif

if [ ! -f license.txt ]
then
    echo "Freesurfer license file missing."
    exit
fi

# BUILD CONTAINER
if [ -z "$build_location" ]
then
      tmpdir=$(mktemp -d)
else
      tmpdir=$(mktemp -d -p ${build_location})
fi
cwd=$(pwd)

cp -r !(*.sif) $tmpdir/

cd $tmpdir

singularity build --fakeroot $container_fname $deffile
mv $container_fname $cwd/
cd $cwd

rm -rf $tmpdir
