CC=g++ -std=c++0x -I ../leveldb/include -I ../ 

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
           triemap.hpp.gch \
           dmsg.hpp.gch
	$(CC) main.cpp -ldl -llua -lpthread 

dmsg.hpp.gch : dmsg.hpp cyqueue.hpp.gch
	$(CC) dmsg.hpp

cyqueue.hpp.gch : cyqueue.hpp
	$(CC) cyqueue.hpp

triemap.hpp.gch : triemap.hpp
	$(CC) triemap.hpp

tctree.hpp.gch : tctree.hpp
	$(CC) tctree.hpp

ktctree.hpp.gch : ktctree.hpp
	$(CC) ktctree.hpp

kbtree.hpp.gch : kbtree.hpp
	$(CC) kbtree.hpp

utils.hpp.gch : utils.hpp
	$(CC) utils.hpp

graph.hpp.gch : graph.hpp
	$(CC) graph.hpp

matht.hpp.gch : matht.hpp
	$(CC) matht.hpp

dectree.hpp.gch : dectree.hpp
	$(CC) dectree.hpp 

math.hpp.gch : math.hpp
	$(CC) math.hpp

kmeans.hpp.gch : kmeans.hpp
	$(CC) kmeans.hpp

GPU.hpp.gch : GPU.hpp
	$(CC) GPU.hpp

active.hpp.gch : active.hpp bayes.hpp.gch dectree.hpp.gch
	$(CC) active.hpp

bayes.hpp.gch : bayes.hpp
	$(CC) bayes.hpp

vec3.hpp.gch : vec3.hpp math.hpp.gch
	$(CC) vec3.hpp

rand.hpp.gch : rand.hpp
	$(CC) rand.hpp

octree.hpp.gch : octree.hpp vec3.hpp.gch
	$(CC) octree.hpp

language.hpp.gch : language.hpp word2vec.hpp.gch sentree.hpp.gch
	$(CC) language.hpp

sentree.hpp.gch : sentree.hpp
	$(CC) sentree.hpp

ksphere.hpp.gch : ksphere.hpp octree.hpp.gch rand.hpp.gch kmeans.hpp.gch
	$(CC) ksphere.hpp

luapool.hpp.gch : luapool.hpp
	$(CC) luapool.hpp

word2vec.hpp.gch : word2vec.hpp ksphere.hpp.gch
	$(CC) word2vec.hpp

module.hpp.gch : module.hpp luapool.hpp.gch
	$(CC) module.hpp

mempool.hpp.gch : mempool.hpp
	$(CC) mempool.hpp

threadpool.hpp.gch : threadpool.hpp
	$(CC) threadpool.hpp

clean:
	rm *.gch
