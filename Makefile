all:./leveldb/out-static/libleveldb.a \
      ./crfpp/crf_learn
	

./leveldb/out-static/libleveldb.a :
	cd leveldb    && make
./crfpp/crf_learn : 
	cd crfpp && ./configure && make
libatulocher.so : atulocher.cpp
	g++ atulocher.cpp -std=c++0x -I ./include -I ./crfpp -fPIC -shared -o libatulocher.so