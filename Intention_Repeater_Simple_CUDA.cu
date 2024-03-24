/*
Intention Repeater Simple CUDA
by Anthro Teacher, WebGPT and Claude 3 Opus
To compile: nvcc -o Intention_Repeater_Simple_CUDA.exe Intention_Repeater_Simple_CUDA.cu -diag-suppress 177
*/

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <signal.h>
#include <iterator>
#include "picosha2.h"
#include <cuda_runtime.h>
#include <cstdio>

#define ONE_MINUTE 60
#define ONE_HOUR 3600

using namespace std;

// CUDA kernel to perform intention repeating and frequency updating
__global__ void intentionRepeaterKernel(char* intentionMultiplied, unsigned long long int* freq, unsigned long long int amplification, size_t intentionSize) {
    unsigned long long int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < amplification) {
        char* processIntention = intentionMultiplied;
        atomicAdd(freq, 1);
    }
}

// CUDA kernel to perform intention hashing
__global__ void intentionHashingKernel(char* intentionMultiplied, char* intentionHashed, size_t intentionSize) {
    unsigned long long int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intentionSize) {
        // Perform hashing using a device-compatible hash function
        // Here, we just copy the character from intentionMultiplied to intentionHashed
        intentionHashed[i] = intentionMultiplied[i];
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
    std::string intention;
    int numGBToUse = 1;
    
    std::cout << "Intention Repeater Simple CUDA" << endl;
    std::cout << "by Anthro Teacher and WebGPT" << endl << endl;
    
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
    long long unsigned int RAM_SIZE = 1024 * 1024 * 512 * numGBToUse, multiplier = 0, hashMultiplier = 1;

    std::cout << "Loading..." << string(10,' ') << "\r" << flush;

    // Append intention repeatedly to intentionMultiplied until the length of intentionMultiplied >= (1024*1024*1024*numGBToUse/2)
    while (intentionMultiplied.length() < RAM_SIZE)
    {
        intentionMultiplied += intention;
        ++multiplier;
    }

     // Allocate memory on the GPU for intentionMultiplied, intentionHashed, and freq
    char* d_intentionMultiplied;
    unsigned long long int* d_freq;
    cudaMalloc(&d_intentionMultiplied, intentionMultiplied.size());
     cudaMalloc(&d_freq, sizeof(unsigned long long int));

    // Copy intentionMultiplied to the GPU
    cudaMemcpy(d_intentionMultiplied, intentionMultiplied.c_str(), intentionMultiplied.size(), cudaMemcpyHostToDevice);

    if (useHashing == "y" || useHashing == "yes") {
        hashMultiplier--;
    
        intentionHashed = picosha2::hash256_hex_string(intentionMultiplied);
        intentionMultiplied = "";
        while (intentionMultiplied.length() < RAM_SIZE) {
            intentionMultiplied += intentionHashed;
            ++hashMultiplier;
        }

        // Update intentionMultiplied on the GPU
        cudaMemcpy(d_intentionMultiplied, intentionMultiplied.c_str(), intentionMultiplied.size(), cudaMemcpyHostToDevice);
    }

    std::string totalIterations = "0", totalFreq = "0", processIntention = "";
    unsigned long long int freq = 0, seconds = 0, amplification=1000000000;
    int digits = 0, freq_digits = 0;

    while (true)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        while ((std::chrono::duration_cast<std::chrono::seconds>(end - start).count() < 1)) {
            // Set freq to 0 on the GPU
            cudaMemset(d_freq, 0, sizeof(unsigned long long int));

            // Launch the CUDA kernel for intention repeating and frequency updating
            int blockSize = 256;
            int numBlocks = (amplification + blockSize - 1) / blockSize;
            intentionRepeaterKernel<<<numBlocks, blockSize>>>(d_intentionMultiplied, d_freq, amplification, intentionMultiplied.size());

            // Copy the updated freq back to the CPU
            cudaMemcpy(&freq, d_freq, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

            end = std::chrono::high_resolution_clock::now();
        }

        totalFreq = std::to_string(freq * multiplier * hashMultiplier);
        totalIterations = FindSum(totalIterations, totalFreq);
 
        digits = totalIterations.length();
        freq_digits = totalFreq.length();
        ++seconds;
        freq = 0;

        std::cout << "[" + FormatTime(seconds) + "] Repeating:"
                    << " (" << DisplaySuffix(totalIterations, digits - 1, "Iterations")
                    << " / " << DisplaySuffix(totalFreq, freq_digits - 1, "Frequency") << "Hz)"
                    << std::string(5, ' ') << "\r" << std::flush;
    }

    // Free allocated memory on the GPU
    cudaFree(d_intentionMultiplied);
    cudaFree(d_freq);

    return 0;
}