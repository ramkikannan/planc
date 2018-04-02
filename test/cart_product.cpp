#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

template <typename T>
void print_2d_vector(vector<vector<T>> in) {
    for (auto &i : in) {
        for (auto &j : i) {
            cout << j << " ";
        }
        cout << endl;
    }
}

vector<vector<int> > cart_product (const vector<vector<int>>& v) {
    vector<vector<int>> s = {{}};
    for (auto& u : v) {
        vector<vector<int>> r;
        for (auto y : u) {
            for (auto& x : s) {
                r.push_back(x);
                r.back().push_back(y);
            }
            // print_2d_vector<int>(r);
        }
        // cout << "before swapping " << endl;
        // print_2d_vector<int>(s);
        s.swap(r);
        // cout << "after swapping " << endl;
        // print_2d_vector<int>(s);
        // cout << "next iteration" << endl;
    }
    return s;
}

int main () {
    vector<vector<int> > test{{1, 2}, {4, 5, 6}, {8, 9}, {1, 2, 3, 4}};
    vector<vector<int> > test_reverse{{1, 2, 3, 4}, {8, 9}, {4, 5, 6}, {1, 2}};
    vector<vector<int> > res = cart_product(test);
    print_2d_vector(res);
    // for (size_t i = 0; i < res.size(); i++) {
    //     for (size_t j = 0; j < res[i].size(); j++) {
    //         cout << res[i][j] << "\t";
    //     }
    //     cout << std::endl;
    // }
    // std::cout << "row major order" << std::endl;
    // vector<vector<int> > res_reverse = cart_product(test);

    // for (size_t i = 0; i < res_reverse.size(); i++) {
    //     size_t res_size = res_reverse[i].size();
    //     for (size_t j = 0; j < res_size; j++) {
    //         cout << res[i][res_size - 1 - j] << "\t";
    //     }
    //     cout << std::endl;
    // }
    return 0;
}