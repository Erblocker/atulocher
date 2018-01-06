PREFIX?=/usr/local
LIBDIR=$(PREFIX)/lib
BINDIR=$(PREFIX)/bin
INCLUDEDIR=$(PREFIX)/include

all:./leveldb/out-static/libleveldb.a \
     atu-inc
	
./leveldb/out-static/libleveldb.a :
	cd leveldb    && make
atu-inc:
	cd atulocher  && make

install:
	@mkdir -p $(INCLUDEDIR)/atulocher
	cp -r atulocher $(INCLUDEDIR)/
	
uninstall:
	rm -rvf $(INCLUDEDIR)/atulocher
