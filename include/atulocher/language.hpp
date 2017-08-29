#ifndef atulocher_language
#define atulocher_language
#include <map>
#include <string>
#include <vector>
#include <set>
#include "word2vec.hpp"
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
    virtual void sentFormat(const std::vector<std::string> & word,
      sentence * sen
    ){
      
    }
  };
}
#endif