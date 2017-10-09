all:./leveldb/out-static/libleveldb.a \
      ./crfpp/crf_learn
	

./leveldb/out-static/libleveldb.a :
	cd leveldb    && make
./crfpp/crf_learn : 
	cd crfpp && ./configure && make