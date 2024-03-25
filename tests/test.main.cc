

#include <string>
#include <iostream>

using std::string;
using std::function;
using std::runtime_error;

using std::cout;

void runtest(const std::string &description, function<bool()> test) {

    if (test()) {
        throw runtime_error(description);
    }

    cout << description << ": Passed";
}


int main() {





    return 0;
}