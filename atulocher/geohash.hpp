#ifndef atulocher_geohash
#define atulocher_geohash
#include <vector>
#include <bitset>
#include <exception>
namespace atulocher {
    namespace geohash{
        class vectorError:public std::exception{};
        void get(
            char * str,
            int len,
            const std::vector<double> & vec,
            const std::vector<double> & beg,
            const std::vector<double> & area,
            char Tr='1',char Fa='0'
        ){
            if( vec.size()!=area.size() ||
                vec.size()!=beg.size()  ||
                vec.size()==0
            ){
                throw vectorError();
            }
            std::vector<double> begin =beg;
            std::vector<double> length=area;

            int size=vec.size();
            int ptr=0;
            int i=0;
            for(;i<len;i++){
                auto half=length[i]/2.0d;
                auto center=begin[i]+half;
                if(vec[i]>center){
                    str[i]=Tr;
                    begin[i]+=half;
                }else{
                    str[i]=Fa;
                }
                length[i]=half;
                if(ptr==size)
                    ptr=0;
                else
                    ++ptr;
            }
            str[len]='\0';
        }
    }
}
#endif
