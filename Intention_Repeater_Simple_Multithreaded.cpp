#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <signal.h>
#include <iterator>
#include <atomic>
#include <mutex>
#include <algorithm>
#include "picosha2.h"
#include <cstring>

#define ONE_MINUTE 60
#define ONE_HOUR 3600
#define NUM_THREADS 8

using namespace std;

atomic<bool> interrupted(false);
atomic<unsigned long long> freqs[NUM_THREADS];
mutex io_mutex;

void signalHandler(int signum)
{
    cout << "\nInterrupt signal (" << signum << ") received.\n";
    interrupted = false;
}

string MultiplyStrings(const string &num1, const string &num2)
{
    int len1 = num1.size();
    int len2 = num2.size();
    vector<int> result(len1 + len2, 0);

    for (int i = len1 - 1; i >= 0; --i)
    {
        for (int j = len2 - 1; j >= 0; --j)
        {
            int mul = (num1[i] - '0') * (num2[j] - '0');
            int sum = mul + result[i + j + 1];

            result[i + j + 1] = sum % 10;
            result[i + j] += sum / 10;
        }
    }

    string resultStr;
    for (int num : result)
    {
        if (!(resultStr.empty() && num == 0))
        {
            resultStr.push_back(num + '0');
        }
    }

    return resultStr.empty() ? "0" : resultStr;
}

void ProcessIntention(int threadId, string intentionMultiplied)
{
    while (!interrupted)
    {
        string process_intention;
        unsigned long long localFreq = 0;
        auto start = chrono::high_resolution_clock::now();
        auto end = start + chrono::seconds(1);
        while (chrono::high_resolution_clock::now() < end)
        {
            // Replace with actual processing if needed
            localFreq++;
            process_intention = intentionMultiplied;
        }
        freqs[threadId] = localFreq;
    }
}

std::string FormatTime(long long int seconds_elapsed)
{
    int hours = seconds_elapsed / ONE_HOUR;
    int minutes = (seconds_elapsed % ONE_HOUR) / ONE_MINUTE;
    int seconds = seconds_elapsed % ONE_MINUTE;

    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << hours << ":"
       << std::setw(2) << std::setfill('0') << minutes << ":"
       << std::setw(2) << std::setfill('0') << seconds;

    return ss.str();
}

std::string DisplaySuffix(std::string num, int power, std::string designator)
{
    std::string suffix_array = designator == "Iterations" ? " kMBTqQsSOND" : " kMGTPEZYR";
    long long unsigned int index = power / 3;
    char suffix = index < suffix_array.length() ? suffix_array[index] : ' ';
    std::string result = num.substr(0, power % 3 + 1) + "." + num.substr(power % 3 + 1, 3) + suffix;
    return result;
}

// Utility function to find the sum of two numbers represented as a string in CPP
std::string FindSum(std::string a, std::string b)
{

    std::vector<int> finalsum; // Stores the final sum of two number

    int carry = 0; // Stores carry at each stage of calculation

    /* Step 1 starts here */

    int i = a.size() - 1,
        j = b.size() - 1; // Start adding from lowest significant bit
    while ((i >= 0) && (j >= 0))
    {                                                // Loop until either of number exhausts first
        int x = (a[i] - '0') + (b[j] - '0') + carry; // Calculate the sum of digit in final sum by adding
                                                     // respective digits with previous carry.
        finalsum.push_back(x % 10);                  // Store the respective digit of the final sum in a vector.
        carry = x / 10;                              // update the carry. The carry for next step is the
                                                     // remaining number after forming the digit of final sum.
        i--;                                         // Move one step towards the left in both the string(numbers)
        j--;
    }
    /*  Step 2 starts here */

    while (i >= 0)
    {                                 // If the number 1 was greater than number 2, then there must
                                      // be some digits to be taken care off.
        int x = (a[i] - '0') + carry; // Add the remaining digits to the carry one
                                      // by one and store the unit digit.
        finalsum.push_back(x % 10);
        carry = x / 10; // update the carry from each step.
        i--;
    }
    /* Step 3 starts here */

    while (j >= 0)
    {                                 // If the number 2 was greater than number 1, then there must
                                      // be some digits to be taken care off.
        int x = (b[j] - '0') + carry; // Add the remaining digits to the carry one
                                      // by one and store the unit digit.
        finalsum.push_back(x % 10);
        carry = x / 10; // update the carry from each step.
        j--;
    }
    /* Step 4 starts here */

    while (carry)
    {                                   // If after finishing addition of the two numbers, if there is
                                        // still carry/leftover then we need to take it into the final
                                        // sum.
        finalsum.push_back(carry % 10); // Store digit one by one.
        carry = carry / 10;             // Reduce carry
    }
    /* Step 5 starts here */
    std::stringstream final_iter;
    // Since vector pushes value at last, the most significant digits starts at
    // the end of the vector. Thus print reverse.

    std::copy(finalsum.rbegin(), finalsum.rend(), std::ostream_iterator<int>(final_iter, ""));

    return final_iter.str();
}

void print_help()
{
    cout << "Intention Repeater Simple (Multithreaded) by Anthro Teacher." << endl;
    cout << "Repeats your intention millions of times per second " << endl;
    cout << "in computer memory, to aid in manifestation." << endl;
    cout << "Optional Flags:" << endl;
    cout << " a) --intent or -i, example: --intent \"I am Love.\" [The Intention]" << endl;
    cout << " b) --imem or -m, example: --imem 2 [GB of RAM to Use]" << endl;
    cout << "    --imem 0 to disable Intention Multiplying" << endl;
    cout << " c) --dur or -d, example: --dur 00:01:00 [Running Duration HH:MM:SS]" << endl;
    cout << " d) --hashing or -h, example: --hashing y [Use Hashing]" << endl;
    cout << " e) --help or -? [This help]" << endl;
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);

    thread threads[NUM_THREADS];

    string intention, param_intent = "X", param_imem = "X", param_duration = "INFINITY", param_hashing = "X", useHashing;
    int numGBToUse = 1;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-?") || !strcmp(argv[i], "--help") || !strcmp(argv[i], "/?"))
        {
            print_help();
            exit(EXIT_SUCCESS);
        }
        else if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--intent"))
        {
            param_intent = argv[i + 1];
        }
        else if (!strcmp(argv[i], "-m") || !strcmp(argv[i], "--imem"))
        {
            param_imem = argv[i + 1];
        }
        else if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--dur"))
        {
            param_duration = argv[i + 1];
        }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--hashing"))
        {
            param_hashing = argv[i + 1];
        }
    }

    std::cout << "Intention Repeater Simple (Multithreaded)" << endl;
    std::cout << "by Anthro Teacher and WebGPT" << endl
              << endl;

    if (param_intent == "X")
    {
        while (!interrupted)
        { // Infinite loop
            std::cout << "Enter your Intention: ";
            std::getline(std::cin, intention);

            // Check if the intention string is not empty
            if (!intention.empty())
            {
                break; // Exit the loop if an intention has been entered
            }

            // Optional: Inform the user that the intention cannot be empty
            std::cout << "The intention cannot be empty. Please try again.\n";
        }
    }
    else
    {
        intention = param_intent;
    }

    if (param_imem == "X")
    {
        cout << "GB RAM to Use [Default 1]: ";
        string input;
        getline(cin, input);
        if (!input.empty())
        {
            numGBToUse = stoi(input);
        }
    }
    else
    {
        numGBToUse = stoi(param_imem);
    }

    if (param_hashing == "X")
    {
        cout << "Use Hashing (y/N): ";
        getline(cin, useHashing);
        transform(useHashing.begin(), useHashing.end(), useHashing.begin(), ::tolower);
    }
    else
    {
        useHashing = param_hashing;
    }

    std::string intentionMultiplied, intentionHashed = "";
    long long unsigned int ramSize = 1024 * 1024 * 1024 * numGBToUse / 2, multiplier = 0, hashMultiplier = 0;

    std::cout << "Loading..." << string(10, ' ') << "\r" << flush;

    if (numGBToUse > 0)
    {
        while (intentionMultiplied.length() < ramSize)
        {
            intentionMultiplied += intention;
            ++multiplier;
        }
    }
    else
    {
        intentionMultiplied = intention;
        multiplier = 1;
    }

    if (useHashing == "y" || useHashing == "yes")
    {
        intentionHashed = picosha2::hash256_hex_string(intentionMultiplied);
        intentionMultiplied.clear();
        if (numGBToUse > 0)
        {
            while (intentionMultiplied.length() < ramSize)
            {
                intentionMultiplied += intentionHashed;
                ++hashMultiplier;
            }
        }
        else
        {
            intentionMultiplied = intentionHashed;
            hashMultiplier = 1;
        }
    }

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        threads[i] = thread(ProcessIntention, i, intentionMultiplied);
    }

    unsigned long long freq = 0, seconds = 0;
    string totalIterations = "0", totalFreq = "0";

    while (!interrupted)
    {
        this_thread::sleep_for(chrono::seconds(1)); // Wait for 1 second

        for (int i = 0; i < NUM_THREADS; ++i)
        {
            freq += freqs[i]; // Sum all frequencies
            freqs[i] = 0;     // Reset for the next count
        }

        totalFreq = MultiplyStrings(to_string(freq), to_string(multiplier));
        totalFreq = MultiplyStrings(totalFreq, to_string(hashMultiplier));
        totalIterations = FindSum(totalIterations, totalFreq);
        int digits = totalIterations.length();
        int freq_digits = totalFreq.length();
        freq = 0;

        lock_guard<mutex> lock(io_mutex);

        cout << "[" << FormatTime(++seconds) << "] Repeating:"
             << " (" << DisplaySuffix(totalIterations, digits - 1, "Iterations")
             << " / " << DisplaySuffix(totalFreq, freq_digits - 1, "Frequency") << "Hz)"
             << string(5, ' ') << "\r" << flush;
        if (param_duration == FormatTime(seconds))
        {
            interrupted = true;
        }
    }

    for (auto &th : threads)
    {
        if (th.joinable())
        {
            th.join();
        }
    }

    return 0;
}