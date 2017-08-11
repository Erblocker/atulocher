include/atulocher/ncnn : ncnn
	cd ncnn && make
	ln -s $(PWD)/ncnn/build-raspberry-armv7 include/atulocher/ncnn