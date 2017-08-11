#ifndef atulocher_language
#define atulocher_language
#include <map>
#include <string>
#include <vector>
#include <set>
namespace atulocher{
  class sentence{
    class element{
      sentence * subordinate;
      std::vector<std::string> modifiers;
      std::string word;
    }
    subject,
    predicate,
    object;
  };
  class langsolv{
    std::map<std::string,double>
      n_phrase_ext,
      adj_phrase_ext,
      adv_phrase_ext,
      s_p_link,
      p_o_link,
      sent_link;
    public:
    langsolv(const char * path){}
    virtual void getTag(const std::string & word,std::set<std::string> & tag){
      
    }
    virtual bool isxc(const std::string & it){
      if(n_phrase_ext.find(it)   !=n_phrase_ext.end()   )return true;
      if(adj_phrase_ext.find(it) !=adj_phrase_ext.end() )return true;
      if(adv_phrase_ext.find(it) !=adv_phrase_ext.end() )return true;
      if(s_p_link.find(it)       !=s_p_link.end()       )return true;
      if(p_o_link.find(it)       !=p_o_link.end()       )return true;
      if(sent_link.find(it)      !=sent_link.end()      )return true;
      return false;
    }
    virtual void cutsent(const std::vector<std::string> & word,
      sentence * sen
    ){
      int i=0,j;
      int s=0,p=0,o=0;
      std::vector<std::set<std::string> > tags(word.size());
      std::vector<std::pair<std::string,double> > as,ap,ao;
      
      for(auto it:word){
        getTag(it,tags[i]);//获取词性
        i++;
      }
      
      i=0;
      for(auto it:word){
        
        if(isxc(it))goto foreach_end;
        //副词不做句子成分
        
        if(tags[i].find("n")!=tags[i].end()){
          //可能是主语
          if(tags[i+1].find("v"))
          double p=1.0d/tags[i].size();
          double pa=1.0d,ps=1.0d;
          for(j=i;j>=0;j--){
            pa/=2.0d;
          }
          pa=1.0d
          for(j=i;j<tags.size();j++){
            if(adj_phrase_ext.find(it) !=adj_phrase_ext.end()){
              p+=pa;
            }else
            if(tags[i].find("adj")!=tags[i].end()){
              p+=pa;
            }else
            if(){
              
            }
            pa/=2.0d;
          }
        }else
        if(tags[i].find("v")!=tags[i].end()){
          //可能是谓语
          for(j=i;j>s;j--){
          
          }
          for(j=i;j<tags.size();j++){
          
          }
        }else
        if(tags[i].find("adj")!=tags[i].end()){
          //可能是谓语
          for(j=i;j>s;j--){
          
          }
          for(j=i;j<tags.size();j++){
          
          }
        }
        
        foreach_end:
        i++;
      }
    }
  };
}
#endif