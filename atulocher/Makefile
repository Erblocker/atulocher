all : atulocher
	
atulocher :NN.hpp.gch \
           module.hpp.gch \
           language.hpp.gch \
           active.hpp.gch
	g++ main.cpp -std=c++0x -ldl -llua -lpthread \
	-I ../leveldb/include \
	../leveldb/out-static/libleveldb.a
math.hpp.gch : math.hpp
	g++ math.hpp -std=c++0x
kmeans.hpp.gch : kmeans.hpp
	g++ kmeans.hpp -std=c++0x
NN_layer.hpp.gch : NN_layer.hpp threadpool.hpp.gch mempool.hpp.gch
	g++ NN_layer.hpp -std=c++0x
NN_MCPU.hpp.gch : NN_MCPU.hpp NN_layer.hpp.gch
	g++ NN_MCPU.hpp -std=c++0x
GPU.hpp.gch : GPU.hpp
	g++ GPU.hpp -std=c++0x
NN_GPU.hpp.gch : NN_GPU.hpp NN_layer.hpp.gch GPU.hpp.gch
	g++ NN_GPU.hpp -std=c++0x
NN.hpp.gch : NN.hpp NN_MCPU.hpp.gch
	g++ NN.hpp -std=c++0x
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
	g++ sentree.hpp -std=c++0x
db.hpp.gch : db.hpp
	g++ db.hpp -std=c++0x -I ../leveldb/include
ksphere.hpp.gch : ksphere.hpp octree.hpp.gch rand.hpp.gch kmeans.hpp.gch
	g++ ksphere.hpp -std=c++0x -I ../leveldb/include
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
