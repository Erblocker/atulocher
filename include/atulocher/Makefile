all : atulocher
	
atulocher :NN.hpp.gch \
           language.hpp.gch \
           luapool.hpp.gch \
           word2vec.hpp.gch
	g++ main.cpp -std=c++0x -ldl -llua -lpthread
math.hpp.gch : math.hpp
	g++ math.hpp -std=c++0x
NN.hpp.gch : NN.hpp
	g++ NN.hpp -std=c++0x
vec3.hpp.gch : vec3.hpp math.hpp.gch
	g++ vec3.hpp -std=c++0x
rand.hpp.gch : rand.hpp
	g++ rand.hpp -std=c++0x
octree.hpp.gch : octree.hpp vec3.hpp.gch
	g++ octree.hpp -std=c++0x
language.hpp.gch : language.hpp
	g++ language.hpp -std=c++0x
ksphere.hpp.gch : ksphere.hpp octree.hpp.gch rand.hpp.gch
	g++ ksphere.hpp -std=c++0x
luapool.hpp.gch : luapool.hpp
	g++ luapool.hpp -std=c++0x
word2vec.hpp.gch : word2vec.hpp ksphere.hpp.gch
	g++ word2vec.hpp -std=c++0x