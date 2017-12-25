all : atulocher
	
atulocher :utils.hpp.gch \
           module.hpp.gch \
           matht.hpp.gch \
           graph.hpp.gch \
           language.hpp.gch \
           active.hpp.gch \
           tctree.hpp.gch \
           ktctree.hpp.gch \
           kbtree.hpp.gch \
           dmsg.hpp.gch
	g++ main.cpp -std=c++0x -ldl -llua -lpthread 

dmsg.hpp.gch : dmsg.hpp cyqueue.hpp.gch
	g++ dmsg.hpp -std=c++0x

cyqueue.hpp.gch : cyqueue.hpp
	g++ cyqueue.hpp -std=c++0x

tctree.hpp.gch : tctree.hpp
	g++ tctree.hpp -std=c++0x

ktctree.hpp.gch : ktctree.hpp
	g++ ktctree.hpp -std=c++0x

kbtree.hpp.gch : kbtree.hpp
	g++ kbtree.hpp -std=c++0x

utils.hpp.gch : utils.hpp
	g++ utils.hpp -std=c++0x -I  ../leveldb/include

graph.hpp.gch : graph.hpp
	g++ graph.hpp -std=c++0x -I  ../leveldb/include

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

active.hpp.gch : active.hpp bayes.hpp.gch dectree.hpp.gch
	g++ active.hpp -std=c++0x -I  ../leveldb/include

bayes.hpp.gch : bayes.hpp
	g++ bayes.hpp -std=c++0x

vec3.hpp.gch : vec3.hpp math.hpp.gch
	g++ vec3.hpp -std=c++0x

rand.hpp.gch : rand.hpp
	g++ rand.hpp -std=c++0x

octree.hpp.gch : octree.hpp vec3.hpp.gch
	g++ octree.hpp -std=c++0x

language.hpp.gch : language.hpp word2vec.hpp.gch sentree.hpp.gch
	g++ language.hpp -std=c++0x

sentree.hpp.gch : sentree.hpp
	g++ sentree.hpp -std=c++0x -I ../ 

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
