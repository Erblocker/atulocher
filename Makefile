all:./leveldb/out-static/libleveldb.a

./leveldb/out-static/libleveldb.a :
	cd leveldb    && make