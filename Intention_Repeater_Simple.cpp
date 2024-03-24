/*
Intention Repeater Simple
by Anthro Teacher, WebGPT and Claude 3 Opus
To compile: g++ -O3 -Wall -static Intention_Repeater_Simple.cpp -o Intention_Repeater_Simple.exe
*/

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include "picosha2.h"

using namespace std;
using namespace std::chrono;

const int ONE_MINUTE = 60;
const int ONE_HOUR = 3600;

string FormatTime(long long seconds) {
    int hours = seconds / ONE_HOUR;
    int minutes = (seconds % ONE_HOUR) / ONE_MINUTE;
    int secs = seconds % ONE_MINUTE;

    ostringstream oss;
    oss << setw(2) << setfill('0') << hours << ":"
        << setw(2) << setfill('0') << minutes << ":"
        << setw(2) << setfill('0') << secs;

    return oss.str();
}

string MultiplyStrings(const string& num1, const string& num2) {
    int len1 = num1.size();
    int len2 = num2.size();
    vector<int> result(len1 + len2, 0);

    for (int i = len1 - 1; i >= 0; --i) {
        for (int j = len2 - 1; j >= 0; --j) {
            int mul = (num1[i] - '0') * (num2[j] - '0');
            int sum = mul + result[i + j + 1];

            result[i + j + 1] = sum % 10;
            result[i + j] += sum / 10;
        }
    }

    string resultStr;
    for (int num : result) {
        if (!(resultStr.empty() && num == 0)) {
            resultStr.push_back(num + '0');
        }
    }

    return resultStr.empty() ? "0" : resultStr;
}

string DisplaySuffix(const string& num, int power, const string& designator) {
    const string suffixArray = designator == "Iterations" ? " kMBTqQsSOND" : " kMGTPEZYR";
    size_t index = power / 3;
    char suffix = index < suffixArray.length() ? suffixArray[index] : ' ';
    string result = num.substr(0, power % 3 + 1) + "." + num.substr(power % 3 + 1, 3) + suffix;
    return result;
}

string FindSum(const string& a, const string& b) {
    string result;
    int carry = 0;

    int i = a.size() - 1;
    int j = b.size() - 1;

    while (i >= 0 || j >= 0 || carry > 0) {
        int sum = carry;

        if (i >= 0) {
            sum += a[i] - '0';
            --i;
        }

        if (j >= 0) {
            sum += b[j] - '0';
            --j;
        }

        result.push_back(sum % 10 + '0');
        carry = sum / 10;
    }

    reverse(result.begin(), result.end());
    return result;
}

int main() {
    string intention;
    int numGBToUse = 1;

    cout << "Intention Repeater Simple" << endl;
    cout << "by Anthro Teacher, WebGPT and Claude 3 Opus" << endl << endl;

    while (intention.empty()) {
        cout << "Enter your Intention: ";
        getline(cin, intention);
    }

    cout << "GB RAM to Use [Default 1]: ";
    string input;
    getline(cin, input);
    if (!input.empty()) {
        numGBToUse = stoi(input);
    }

    cout << "Use Hashing (y/N): ";
    string useHashing;
    getline(cin, useHashing);
    transform(useHashing.begin(), useHashing.end(), useHashing.begin(), ::tolower);

    string intentionMultiplied, intentionHashed;
    size_t ramSize = 1024ULL * 1024 * 1024 * numGBToUse / 2;
    size_t multiplier = 0, hashMultiplier = 1;

    cout << "Loading..." << string(10, ' ') << "\r" << flush;

    while (intentionMultiplied.length() < ramSize) {
        intentionMultiplied += intention;
        ++multiplier;
    }

    if (useHashing == "y" || useHashing == "yes") {
        intentionHashed = picosha2::hash256_hex_string(intentionMultiplied);
        intentionMultiplied.clear();
        while (intentionMultiplied.length() < ramSize) {
            intentionMultiplied += intentionHashed;
            ++hashMultiplier;
        }
    }

    string totalIterations = "0", totalFreq = "0";
    unsigned long long freq = 0, seconds = 0;

    while (true) {
        auto start = high_resolution_clock::now();
        auto end = start + chrono::duration_cast<chrono::seconds>(chrono::seconds(1));

        while (high_resolution_clock::now() < end) {
            volatile string processIntention = intentionMultiplied; // Prevent optimization
            freq++;
        }

        totalFreq = MultiplyStrings(to_string(freq), to_string(multiplier));
        totalFreq = MultiplyStrings(totalFreq, to_string(hashMultiplier));
        totalIterations = FindSum(totalIterations, totalFreq);

        int digits = totalIterations.length();
        int freqDigits = totalFreq.length();
        ++seconds;
        freq = 0;

        cout << "[" + FormatTime(seconds) + "] Repeating:"
             << " (" << DisplaySuffix(totalIterations, digits - 1, "Iterations")
             << " / " << DisplaySuffix(totalFreq, freqDigits - 1, "Frequency") << "Hz)"
             << string(5, ' ') << "\r" << flush;
    }

    return 0;
}