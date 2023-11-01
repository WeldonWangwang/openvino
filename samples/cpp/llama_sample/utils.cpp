#include <iterator>
#include <memory>
#include <sstream>
#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "utils.hpp"

std::map<std::string, int32_t> llama_tokenizer::read_json(const std::string & fname) {
    std::map<std::string, int32_t> result;
    auto replace = [](std::string & str, const std::string & needle, const std::string & replacement){
        size_t pos = 0;
        while ((pos = str.find(needle, pos)) != std::string::npos) {
            str.replace(pos, needle.length(), replacement);
            pos += replacement.length();
        }        
    };
    // read file into string
    std::string json;
    {
        std::ifstream ifs(fname);
        if (!ifs) {
            fprintf(stderr, "Failed to open %s\n", fname.c_str());
            exit(1);
        }

        json = std::string((std::istreambuf_iterator<char>(ifs)),
                (std::istreambuf_iterator<char>()));
    }
    if (json[0] != '{') {
        return result;
    }

    // parse json
    {
        bool has_key  = false;
        bool in_token = false;

        std::string str_key = "";
        std::string str_val = "";

        int n = json.size();
        for (int i = 1; i < n; ++i) {
            if (!in_token) {
                if (json[i] == ' ') continue;
                if (json[i] == '"') {
                    in_token = true;
                    continue;
                }
            } else {
                if (json[i] == '\\' && i+1 < n) {
                    if (has_key == false) {
                        str_key += json[i];
                    } else {
                        str_val += json[i];
                    }
                    ++i;
                } else if (json[i] == '"') {
                    if (has_key == false) {
                        has_key = true;
                        ++i;
                        while (json[i] == ' ') ++i;
                        ++i; // :
                        while (json[i] == ' ') ++i;
                        if (json[i] != '\"') {
                            while (json[i] != ',' && json[i] != '}') {
                                str_val += json[i++];
                            }
                            has_key = false;
                        } else {
                            in_token = true;
                            continue;
                        }
                    } else {
                        has_key = false;
                    }

                    replace(str_key, "\\u0120", " " ); // \u0120 -> space
                    replace(str_key, "\\u010a", "\n"); // \u010a -> new line
                    replace(str_key, "\\\"",    "\""); // \\\"   -> "

                    try {
                        result[str_key] = std::stoi(str_val);
                    } catch (...) {
                        //fprintf(stderr, "%s: ignoring key '%s' with value '%s'\n", fname.c_str(), str_key.c_str(), str_val.c_str());

                    }
                    str_key = "";
                    str_val = "";
                    in_token = false;
                    continue;
                }
                if (has_key == false) {
                    str_key += json[i];
                } else {
                    str_val += json[i];
                }
            }
        }
    }

    return result;
}

llama_tokenizer::llama_tokenizer(const std::string & fname) {
    vocab.token_to_id = read_json(fname);
    for (const auto & kv : vocab.token_to_id) {
        vocab.id_to_token[kv.second] = kv.first;
    }
}
#define MAX_TOKEN_LEN 18
std::vector<int> llama_tokenizer::tokenizer(std::string prompt_text) {
    std::vector<llama_vocab::id> res;
    std::vector<int> score;
    std::vector<llama_vocab::id> prev;
    int len = prompt_text.length();
    score.resize(len + 1);
    prev.resize(len + 1);

    for (int i = 0; i < len; i++) {
        int max_len = std::min(len - i, MAX_TOKEN_LEN);
        for (int sub_len = 1; sub_len <= max_len; sub_len++) {
            auto sub = prompt_text.substr(i, sub_len);
            if (sub.rfind(" ", 0) == 0) {
                sub.replace(0, 1, "▁");
            }
            auto token = vocab.token_to_id.find(sub);
            if (token != vocab.token_to_id.end()) {
                int token_score = sub.length() * sub.length();
                int local_score = score[i] + token_score;
                int next = i + sub_len;
                if (score[next] < local_score) {
                    score[next] = local_score;
                    prev[next] = (*token).second;
                }
            }
        }
    }

    int i = len;
    while (i > 0) {
        llama_vocab::id token_id = prev[i];
        res.push_back(token_id);
        auto token = (*vocab.id_to_token.find(token_id)).second;
        auto token_length = token.length();
        if (token.rfind("▁", 0) == 0) {
            token_length = token.length() - 2;
        }
        i -= token_length;
    }

    // Pieces are in reverse order so correct that
    std::reverse(res.begin(), res.end());

    return res;
}

std::string llama_tokenizer::tokenizer_decode(int id) {
    auto token = vocab.id_to_token.find(id);
    if (token != vocab.id_to_token.end()) {
        auto token_text = (*token).second;
        if (token_text.rfind("▁", 0) == 0) {
            token_text.replace(0, 3, " ");
        }
        return token_text;
    } else {
        std::cout << "Failed to decode " << id << std::endl;
        return "";
    }
}

int llama_tokenizer::get_llama_token_eos() {
    return 2;
}

int llama_tokenizer::get_llama_n_vocab() {
    return 32000;
}