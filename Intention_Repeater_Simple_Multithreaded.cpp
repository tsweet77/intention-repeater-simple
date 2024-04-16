/*
Intention Repeater Simple (Multithreaded)
by Anthro Teacher, WebGPT and Claude 3 Opus
To compile: g++ -O3 -Wall -static Intention_Repeater_Simple_Multithreaded.cpp -o Intention_Repeater_Simple_Multithreaded.exe -lz
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
#include <atomic>
#include <mutex>
#include <algorithm>
#include "picosha2.h"
#include <cstring>
#include "zlib.h"

#define ONE_MINUTE 60
#define ONE_HOUR 3600
#define NUM_THREADS 8

using namespace std;

string VERSION = "v1.2";

atomic<bool> interrupted(false);
atomic<unsigned long long> freqs[NUM_THREADS];
mutex io_mutex;

void signalHandler(int signum)
{
    //cout << "\nInterrupt signal (" << signum << ") received.\n";
    interrupted = true;
}

std::string compressMessage(const std::string &message)
{
    z_stream zs;
    memset(&zs, 0, sizeof(zs));

    if (deflateInit(&zs, Z_DEFAULT_COMPRESSION) != Z_OK)
    {
        return ""; // Compression initialization failed
    }

    zs.next_in = (Bytef *)message.data();
    zs.avail_in = message.size();

    std::string compressed;
    char outbuffer[32768]; // Output buffer
    int ret;
    do
    {
        zs.next_out = reinterpret_cast<Bytef *>(outbuffer);
        zs.avail_out = sizeof(outbuffer);

        ret = deflate(&zs, Z_FINISH);

        if (compressed.size() < zs.total_out)
        {
            compressed.append(outbuffer, zs.total_out - compressed.size());
        }
    } while (ret == Z_OK);

    deflateEnd(&zs);

    if (ret != Z_STREAM_END)
    {
        return ""; // Compression failed
    }

    return compressed;
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
        string processIntention;
        processIntention.reserve(intentionMultiplied.size() + 20); // Adjust based on expected size
        unsigned long long localFreq = 0;
        auto start = chrono::high_resolution_clock::now();
        auto end = start + chrono::seconds(1);
        while (chrono::high_resolution_clock::now() < end)
        {
            // Clear previous value and reuse the allocated space
            processIntention.clear();
            // Append the fixed part and the changing part
            processIntention.append(intentionMultiplied);
            processIntention.append(to_string(localFreq));
            // Replace with actual processing if needed
            localFreq++;
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
    cout << "Intention Repeater Simple by Anthro Teacher." << endl;
    cout << "Repeats your intention millions of times per second " << endl;
    cout << "in computer memory, to aid in manifestation." << endl;
    cout << "Optional Flags:" << endl;
    cout << " a) --intent or -i, example: --intent \"I am Love.\" [The Intention]" << endl;
    cout << " b) --imem or -m, example: --imem 2 [GB of RAM to Use]" << endl;
    cout << "    --imem 0 to disable Intention Multiplying" << endl;
    cout << " c) --dur or -d, example: --dur 00:01:00 [Running Duration HH:MM:SS]" << endl;
    cout << " d) --hashing or -h, example: --hashing y [Use Hashing]" << endl;
    cout << " e) --compress or -c, example: --compress y [Use Compression]" << endl;
    cout << " f) --file or -f, example: --file \"intentions.txt\" [File to Read Intentions From]" << endl;
    cout << " g) --help or -? [This help]" << endl;
}

void readFileContents(const std::string &filename,
                      std::string &intention_file_contents)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "File not found" << std::endl;
        std::exit(EXIT_FAILURE); // Terminate the program
    }

    std::ostringstream buffer;
    char ch;
    while (file.get(ch))
    {
        if (ch != '\0')
        {
            buffer.put(ch);
        }
    }

    intention_file_contents = buffer.str();
    file.close();
}

int main(int argc, char **argv)
{
    std::cout << "Intention Repeater Simple (Multithreaded) " << VERSION << endl;
    std::cout << "by Anthro Teacher and WebGPT" << endl
              << endl;

    signal(SIGINT, signalHandler);

    thread threads[NUM_THREADS];

    string intention = "", param_intent = "X", param_imem = "X", param_duration = "INFINITY", param_hashing = "X";
    string useHashing, param_compress = "X", useCompression, param_file = "X", intention_display = "";
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
        else if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "--compress"))
        {
            param_compress = argv[i + 1];
        }
        else if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--file"))
        {
            param_file = argv[i + 1];
        }
    }

    if (param_file != "X")
    {
        // Open param_intent file and read the full file contents into intention
        readFileContents(param_file, intention);
        intention_display = "Contents of: " + param_file;
    } else
    {
        if (param_intent == "X")
        {
            while (!interrupted)
            {
                std::cout << "Enter your Intention: ";
                if (!std::getline(std::cin, intention))
                {
                    // If getline fails (e.g., due to an interrupt), break out of the loop immediately
                    interrupted.store(true); // Ensure the flag is set if not already
                    break;
                }

                if (!intention.empty())
                {
                    break; // Successfully got an intention, exit the loop
                }
                else if (!interrupted)
                {
                    // Only show the message if we're not interrupted
                    std::cout << "The intention cannot be empty. Please try again.\n";
                }
            }
            intention_display = intention;
        }
        else
        {
            intention = param_intent;
            intention_display = param_intent;
        }
    }

        if (!interrupted)
    {
        if (param_imem == "X")
        {
            std::cout << "GB RAM to Use [Default 1]: ";
            string input;
            if (!std::getline(std::cin, input))
            {
                // If getline fails due to interruption
                interrupted.store(true); // Ensure the flag is properly set
                if (interrupted)
                {
                    // std::cerr << "Interrupted. Exiting configuration.\n";
                    return 0; // Exit or handle as necessary
                }
            }

            if (!input.empty())
            {
                try
                {
                    numGBToUse = stoi(input);
                }
                catch (const std::invalid_argument &e)
                {
                    // std::cerr << "Invalid input, using default of 1 GB.\n";
                    numGBToUse = 1;
                }
                catch (const std::out_of_range &e)
                {
                    // std::cerr << "Input out of range, using default of 1 GB.\n";
                    numGBToUse = 1;
                }
            }
        }
        else
        {
            numGBToUse = stoi(param_imem);
        }
    }

    if (!interrupted && param_hashing == "X")
    {
        std::cout << "Use Hashing (y/N): ";
        if (!std::getline(std::cin, useHashing))
        {
            interrupted.store(true);
            if (interrupted)
            {
                // std::cerr << "Interrupted during hashing input. Exiting configuration.\n";
                return 0;
            }
        }
        transform(useHashing.begin(), useHashing.end(), useHashing.begin(), ::tolower);
    }
    else if (!interrupted)
    {
        useHashing = param_hashing;
    }

    if (!interrupted && param_compress == "X")
    {
        std::cout << "Use Compression (y/N): ";
        if (!std::getline(std::cin, useCompression))
        {
            interrupted.store(true);
            if (interrupted)
            {
                // std::cerr << "Interrupted during compression input. Exiting configuration.\n";
                return 0;
            }
        }
        transform(useCompression.begin(), useCompression.end(), useCompression.begin(), ::tolower);
    }
    else if (!interrupted)
    {
        useCompression = param_compress;
    }

    std::string intentionMultiplied, intentionHashed = "";
    long long unsigned int ramSize = 1024 * 1024 * 1024 * numGBToUse / 2, multiplier = 0, hashMultiplier = 0;

    if (!interrupted)
    {
        std::cout << "Loading..." << string(10, ' ') << "\r" << flush;
    }
    else
    {
        return 0;
    }

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
    else
    {
        hashMultiplier = 1;
    }

    if (useCompression == "y" || useCompression == "yes")
    {
        intentionMultiplied = compressMessage(intentionMultiplied);
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

        cout << "[" << FormatTime(++seconds) << "] " << " (" << DisplaySuffix(totalIterations, digits - 1, "Iterations")
             << " / " << DisplaySuffix(totalFreq, freq_digits - 1, "Frequency") << "Hz): " << intention_display
             << string(5, ' ') << "\r" << flush;
        if (param_duration == FormatTime(seconds))
        {
            interrupted = true;
        }

        if (interrupted)
        {
            break;
        }
    }

    for (auto &th : threads)
    {
        if (th.joinable())
        {
            th.join();
        }
    }

    std::cout << endl;
    return 0;
}