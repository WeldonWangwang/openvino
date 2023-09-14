#include <iterator>
#include <memory>
#include <sstream>
#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include <algorithm>

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
};

class llama_tokenizer
{
public:
    llama_tokenizer(const std::string & fname);
    std::map<std::string, int32_t> read_json(const std::string & fname);
    llama_vocab vocab;
    std::vector<int> tokenizer(std::string prompt_text);
    std::string tokenizer_decode(int id);
    int get_llama_token_eos();
    int get_llama_n_vocab();
};