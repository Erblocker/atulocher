all : atulocher
	
atulocher :NN.hpp.gch \
           language.hpp.gch \
           ksphere.hpp.gch
	
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
	g++ language.hpp -std=c++0x -I ../../core
ksphere.hpp.gch : ksphere.hpp octree.hpp.gch rand.hpp.gch
	g++ ksphere.hpp -std=c++0x