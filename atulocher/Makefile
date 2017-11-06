all : atulocher
	
atulocher :utils.hpp.gch \
           module.hpp.gch \
           matht.hpp.gch \
           language.hpp.gch \
           active.hpp.gch \
           dectree.hpp.gch
	g++ main.cpp -std=c++0x -ldl -llua -lpthread 

utils.hpp.gch : utils.hpp
	g++ utils.hpp -std=c++0x -I  ../leveldb/include

matht.hpp.gch : matht.hpp
	g++ matht.hpp -std=c++0x

dectree.hpp.gch : dectree.hpp
	g++ dectree.hpp -std=c++0x -I ../leveldb/include

math.hpp.gch : math.hpp
	g++ math.hpp -std=c++0x

kmeans.hpp.gch : kmeans.hpp
	g++ kmeans.hpp -std=c++0x

GPU.hpp.gch : GPU.hpp
	g++ GPU.hpp -std=c++0x

active.hpp.gch : active.hpp bayes.hpp.gch
	g++ active.hpp -std=c++0x

bayes.hpp.gch : bayes.hpp
	g++ bayes.hpp -std=c++0x

vec3.hpp.gch : vec3.hpp math.hpp.gch
	g++ vec3.hpp -std=c++0x

rand.hpp.gch : rand.hpp
	g++ rand.hpp -std=c++0x

octree.hpp.gch : octree.hpp vec3.hpp.gch
	g++ octree.hpp -std=c++0x

language.hpp.gch : language.hpp word2vec.hpp.gch sentree.hpp.gch
	g++ language.hpp -std=c++0x -I ../crfpp

sentree.hpp.gch : sentree.hpp
	g++ sentree.hpp -std=c++0x -I ../ -I ../crfpp

ksphere.hpp.gch : ksphere.hpp octree.hpp.gch rand.hpp.gch kmeans.hpp.gch
	g++ ksphere.hpp -std=c++0x

luapool.hpp.gch : luapool.hpp
	g++ luapool.hpp -std=c++0x

word2vec.hpp.gch : word2vec.hpp ksphere.hpp.gch
	g++ word2vec.hpp -std=c++0x

module.hpp.gch : module.hpp luapool.hpp.gch
	g++ module.hpp -std=c++0x

mempool.hpp.gch : mempool.hpp
	g++ mempool.hpp -std=c++0x

threadpool.hpp.gch : threadpool.hpp
	g++ threadpool.hpp -std=c++0x

clean:
	rm *.gch
