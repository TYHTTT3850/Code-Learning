#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <cstring>

using namespace std;

/* ===============================
   char* / const char* -> vector
   =============================== */

// C 风格字符串（不包含 '\0'）
vector<char> cstr_to_vector(const char* s) {
    size_t len = strlen(s);
    return vector<char>(s, s + len);
}

// C 风格字符串（包含 '\0'）
vector<char> cstr_to_vector_with_null(const char* s) {
    size_t len = strlen(s);
    return vector<char>(s, s + len + 1);
}

// 指针 + 长度（最高效）
vector<char> ptr_to_vector(const char* p, size_t len) {
    return vector<char>(p, p + len);
}

/* ===============================
   string -> vector
   =============================== */

vector<char> string_to_vector(const string& s) {
    return vector<char>(s.begin(), s.end());
}

vector<char> string_to_vector_cstyle(const string& s) {
    vector<char> v(s.begin(), s.end());
    v.push_back('\0');
    return v;
}

/* ===============================
   vector -> string
   =============================== */

string vector_to_string(const vector<char>& v) {
    return string(v.data(), v.size());
}

/* ===============================
   零拷贝：string_view
   =============================== */

string_view view_from_cptr(const char* p, size_t len) {
    return string_view(p, len);
}

string_view view_from_vector(const vector<char>& v) {
    return string_view(v.data(), v.size());
}

string_view view_from_string(const string& s) {
    return string_view(s);
}

/* ===============================
   主函数测试
   =============================== */

int main() {
    const char* cstr = "Hello C++";
    string str = "Modern C++";
    vector<char> buf = {'A', 'I', '-', 'M', 'L'};

    // char* -> vector
    auto v1 = cstr_to_vector(cstr);
    auto v2 = cstr_to_vector_with_null(cstr);

    // string -> vector
    auto v3 = string_to_vector(str);
    auto v4 = string_to_vector_cstyle(str);

    // vector -> string
    string s1 = vector_to_string(buf);

    // 零拷贝 string_view
    string_view sv1 = view_from_cptr(cstr, strlen(cstr));
    string_view sv2 = view_from_vector(buf);
    string_view sv3 = view_from_string(str);

    cout << "v1 size: " << v1.size() << endl;
    cout << "v2 size (with \\0): " << v2.size() << endl;
    cout << "s1 from vector: " << s1 << endl;

    cout << "sv1: " << sv1 << endl;
    cout << "sv2: " << sv2 << endl;
    cout << "sv3: " << sv3 << endl;

    return 0;
}
