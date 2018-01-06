#!/bin/sh

dir=./disentangle_anim; [ ! -e $dir ] && mkdir -p $dir

for i in `seq 0 9`
do
	png_name="disentangle_img/check_z${i}_*.png";
	gif_name="disentangle_anim/anim_z${i}.gif";
	convert $png_name $gif_name;
done
