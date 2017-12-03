SHELL=/bin/bash
CC=gcc

default: all

.PHONY: clean

demo:
	cd src/demo/ && $(MAKE)
neural_cli:
	cd src && $(MAKE)
test:
	cd src/test && $(MAKE)
profile:
	cd src/debug && $(MAKE)
clean:
	if ! [ -e tmp/ ]; then mkdir tmp/; fi
	if [ -e bin/README ]; then cp bin/README tmp/; fi
	rm -f src/*.o
	rm -f src/demo/*.o
	rm -f src/test/*.o
	rm -f bin/*
	rm -f lib/*
	if [ -e tmp/README ]; then cp tmp/README bin/; fi
	if [ -e tmp/README ]; then cp tmp/README lib/; fi

install:
	cd src && $(MAKE) install
uninstall:
	cd src && $(MAKE) uninstall
all: neural_cli demo
	
        
