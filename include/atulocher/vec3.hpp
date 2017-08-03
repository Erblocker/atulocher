#ifndef atulocher_vec3
#define atulocher_vec3
#include "math.hpp"
#include <math.h>
namespace atulocher{
  template<typename T> class vec3{
    public:
    T x;
    T y;
    T z;
    vec3(){
      x=0.0f;
      y=0.0f;
      z=0.0f;
    }
    vec3(T xt,T yt,T zt){
      x=xt;
      y=yt;
      z=zt;
    }
    void init(T xt,T yt,T zt){
      x=xt;
      y=yt;
      z=zt;
    }
    bool operator==(const vec3<T> &i)const{
      if(x==i.x)
      if(x==i.y)
      if(x==i.z)
      return true;
      return false;
    }
    void operator()(T xt,T yt,T zt){
      init(xt,yt,zt);
    }
    vec3<T>& operator=(const vec3<T> *p){
      x=p->x;
      y=p->y;
      z=p->z;
      return *this;
    }
    vec3<T>& operator=(const vec3<T> &p){
      x=p.x;
      y=p.y;
      z=p.z;
      return *this;
    }
    vec3<T> operator+(const vec3<T> &p)const{
      vec3<T> b;
      b=this;
      b.x+=p.x;
      b.y+=p.y;
      b.z+=p.z;
      return b;
    }
    vec3<T> & operator+=(const vec3<T> &p){
      x+=p.x;
      y+=p.y;
      z+=p.z;
      return *this;
    }
    vec3<T> & operator-=(const vec3<T> &p){
      x-=p.x;
      y-=p.y;
      z-=p.z;
      return *this;
    }
    vec3<T> operator-(const vec3<T> &p)const{
      vec3<T> b;
      b=this;
      b.x-=p.x;
      b.y-=p.y;
      b.z-=p.z;
      return b;
    }
    vec3<T> operator*(const T &p)const{
      return vec3<T>(p*x , p*y , p*z);
    }
    vec3<T> operator/(const T &p)const{
      return vec3<T>(x/p , y/p , z/p);
    }
    vec3<T> operator*(const vec3<T> &i)const{
      return vec3<T>(
        y * i.z - z * i.y,
        z * i.x - x * i.z,
        z * i.y - y * i.x
      );
    }
    T norm()const{
      return sqrt((x*x)+(y*y)+(z*z));
    }
    T invnorm()const{
      return math::invsqrt((x*x)+(y*y)+(z*z));
    }
    T pro(const vec3<T> *p)const{
      return math::sqrt(
        (x*p->x)+
        (y*p->y)+
        (z*p->z)
      );
    }
    T pro(const vec3<T> &p)const{
      return math::sqrt(
        (x*p.x)+
        (y*p.y)+
        (z*p.z)
      );
    }
    void GeoHash(T length,char * str,int l)const{
      vec3<T> v;
      GeoHash(length,str,0,l,&v);
    }
    void GeoHash(T length,char str[],int begin,int end,vec3<T> *zero)const{
      if(begin==end){
        str[begin]=0x00;
        return;
      }
      if(x>zero->x){
        if(y>zero->y){
          if(z>zero->z){
            str[begin]='a';
            zero->x+=length;
            zero->y+=length;
            zero->z+=length;
            GeoHash(length*0.5f,str,begin+1,end,zero);
          }else{
            str[begin]='b';
            zero->x+=length;
            zero->y+=length;
            zero->z-=length;
            GeoHash(length*0.5f,str,begin+1,end,zero);
          }
        }else{
          if(z>zero->z){
            str[begin]='c';
            zero->x+=length;
            zero->y-=length;
            zero->z+=length;
            GeoHash(length*0.5f,str,begin+1,end,zero);
          }else{
            str[begin]='d';
            zero->x+=length;
            zero->y-=length;
            zero->z-=length;
            GeoHash(length*0.5f,str,begin+1,end,zero);
          }
        }
      }else{
        if(y>zero->y){
          if(z>zero->z){
            str[begin]='e';
            zero->x-=length;
            zero->y+=length;
            zero->z+=length;
            GeoHash(length*0.5f,str,begin+1,end,zero);
          }else{
            str[begin]='f';
            zero->x-=length;
            zero->y+=length;
            zero->z-=length;
            GeoHash(length*0.5f,str,begin+1,end,zero);
          }
        }else{
          if(z>zero->z){
            str[begin]='g';
            zero->x-=length;
            zero->y-=length;
            zero->z+=length;
            GeoHash(length*0.5f,str,begin+1,end,zero);
          }else{
            str[begin]='h';
            zero->x-=length;
            zero->y-=length;
            zero->z-=length;
            GeoHash(length*0.5f,str,begin+1,end,zero);
          }
        }
      }
    }
    bool GeoHashDecode(T length,const char * s){
      const char * str=s;
      T ll=length;
      x=0.0f;
      y=0.0f;
      z=0.0f;
      while(*str){
        switch(*str){
          case ('a'):
            x+=ll;
            y+=ll;
            z+=ll;
          break;
          case ('b'):
            x+=ll;
            y+=ll;
            z-=ll;
          break;
          case ('c'):
            x+=ll;
            y-=ll;
            z+=ll;
          break;
          case ('d'):
            x+=ll;
            y-=ll;
            z-=ll;
          break;
          case ('e'):
            x-=ll;
            y+=ll;
            z+=ll;
          break;
          case ('f'):
            x-=ll;
            y+=ll;
            z-=ll;
          break;
          case ('g'):
            x-=ll;
            y-=ll;
            z+=ll;
          break;
          case ('h'):
            x-=ll;
            y-=ll;
            z-=ll;
          break;
          default:
            return false;
          break;
        }
        ll=ll*0.5f;
        str++;
      }
      return true;
    }
  };
}
#endif