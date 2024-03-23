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

#define ONE_MINUTE 60
#define ONE_HOUR 3600
#define NUM_THREADS 8

using namespace std;

atomic<bool> running(true);
atomic<unsigned long long> freqs[NUM_THREADS];
mutex io_mutex;

void signalHandler(int signum) {
    cout << "\nInterrupt signal (" << signum << ") received.\n";
    running = false;
}

void ProcessIntention(int threadId, string intentionMultiplied) {
    while (running) {
        string process_intention;
        unsigned long long localFreq = 0;
        auto start = chrono::high_resolution_clock::now();
        auto end = start + chrono::seconds(1);
        while (chrono::high_resolution_clock::now() < end) {
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
    std::string suffix_array = designator == "Iterations" ? " kMBTqQsSOND" : " kMGTPEZY";
    long long unsigned int index = power / 3;
    char suffix = index < suffix_array.length() ? suffix_array[index] : ' ';
    std::string result = num.substr(0, power % 3 + 1) + "." + num.substr(power % 3 + 1, 3) + suffix;
    return result;
}

// Utility function to find the sum of two numbers represented as a string in
// CPP
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

int main()
{
    signal(SIGINT, signalHandler);

    thread threads[NUM_THREADS];

    std::string intention;
    int numGBToUse = 1;
    while (true)
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

    std::cout << "GB RAM to Use [Default 1]: ";
    std::string input;
    std::getline(std::cin, input);
    if (!input.empty())
    {
        std::istringstream(input) >> numGBToUse;
    }

    std::cout << "Use Hashing (y/N): ";
    std::string useHashing;
    std::getline(cin, useHashing);
    std::transform(useHashing.begin(), useHashing.end(), useHashing.begin(), ::tolower);

    std::string intentionMultiplied, intentionHashed = "";
    long long unsigned int RAM_SIZE = 1024 * 1024 * 1024 * numGBToUse / 2, multiplier = 0, hashMultiplier = 0;

    std::cout << "Loading..." << string(10,' ') << "\r" << flush;

    // Append intention repeatedly to intentionMultiplied until the length of intentionMultiplied >= (1024*1024*1024*numGBToUse/2)
    while (intentionMultiplied.length() < RAM_SIZE)
    {
        intentionMultiplied += intention;
        ++multiplier;
    }

    if (useHashing == "y" || useHashing == "yes")
    {
        intentionHashed = picosha2::hash256_hex_string(intentionMultiplied);
        intentionMultiplied = "";
        while (intentionMultiplied.length() < RAM_SIZE)
        {
            intentionMultiplied += intentionHashed;
            ++hashMultiplier;
        }
        multiplier = multiplier * hashMultiplier;
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads[i] = thread(ProcessIntention, i, intentionMultiplied);
    }

string totalIterations = "0";
    while (running) {
        this_thread::sleep_for(chrono::seconds(1)); // Wait for 1 second

        unsigned long long totalFreq = 0, seconds = 0;
        for (int i = 0; i < NUM_THREADS; ++i) {
            totalFreq += freqs[i]; // Sum all frequencies
            freqs[i] = 0; // Reset for the next count
        }

        string freqStr = to_string(totalFreq * multiplier);
        totalIterations = FindSum(totalIterations, freqStr);
        int digits = totalIterations.length();
        int freq_digits = freqStr.length();

        lock_guard<mutex> lock(io_mutex);
        cout << "[" << FormatTime(++seconds) << "] Repeating: "
             << " (" << DisplaySuffix(totalIterations, digits - 1, "Iterations")
             << " / " << DisplaySuffix(freqStr, freq_digits - 1, "Frequency") << "Hz)"
             << string(5, ' ') << "\r" << flush;
        seconds++;
    }

    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    return 0;
}