/*
Intention Repeater Simple CUDA
by Anthro Teacher, WebGPT and Claude 3 Opus
To compile: nvcc Intention_Repeater_Simple_CUDA.cu -o Intention_Repeater_Simple_CUDA.exe -L/Users/tswee/miniconda3/Library/lib -lz
To run: Intention_Repeater_Simple_CUDA.exe --intent "I am Love." --imem 1 --hashing y --compress n --dur 00:00:10
*/

#include "picosha2.h"
#include "zlib.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <csignal>
#include <atomic>

using namespace std;
using namespace std::chrono;

const int ONE_MINUTE = 60;
const int ONE_HOUR = 3600;

string VERSION = "v1.3";

std::atomic<bool> interrupted(false);

void signalHandler(int signum)
{
    //cout << "\nInterrupt signal (" << signum << ") received.\n";
    interrupted.store(true);
}

// CUDA kernel to perform intention repeating and frequency updating
__global__ void intentionRepeaterKernel(const char *intentionMultiplied, unsigned long long int *freq, size_t intentionSize)
{
    unsigned long long int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intentionSize)
    {
        atomicAdd(freq, 1);
    }
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

string FormatTime(long long seconds)
{
    int hours = seconds / ONE_HOUR;
    int minutes = (seconds % ONE_HOUR) / ONE_MINUTE;
    int secs = seconds % ONE_MINUTE;

    ostringstream oss;
    oss << setw(2) << setfill('0') << hours << ":"
        << setw(2) << setfill('0') << minutes << ":"
        << setw(2) << setfill('0') << secs;

    return oss.str();
}

void print_help()
{
    cout << "Intention Repeater Simple CUDA by Anthro Teacher." << endl;
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

string DisplaySuffix(const string &num, int power, const string &designator)
{
    const string suffixArray = designator == "Iterations" ? " kMBTqQsSOND" : " kMGTPEZYR";
    size_t index = power / 3;
    char suffix = index < suffixArray.length() ? suffixArray[index] : ' ';
    string result = num.substr(0, power % 3 + 1) + "." + num.substr(power % 3 + 1, 3) + suffix;
    return result;
}

string FindSum(const string &a, const string &b)
{
    string result;
    int carry = 0;

    int i = a.size() - 1;
    int j = b.size() - 1;

    while (i >= 0 || j >= 0 || carry > 0)
    {
        int sum = carry;

        if (i >= 0)
        {
            sum += a[i] - '0';
            --i;
        }

        if (j >= 0)
        {
            sum += b[j] - '0';
            --j;
        }

        result.push_back(sum % 10 + '0');
        carry = sum / 10;
    }

    reverse(result.begin(), result.end());
    return result;
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
    std::cout << "Intention Repeater Simple CUDA " << VERSION << endl;
    std::cout << "by Anthro Teacher and WebGPT" << endl
              << endl;

    std::signal(SIGINT, signalHandler);
    string intention = "", param_intent = "X", param_imem = "X", param_duration = "INFINITY", param_hashing = "X";
    string useHashing, useCompression, param_compress = "X", param_file = "X", intention_display = "", file_contents="", intention_value="";
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
        readFileContents(param_file, file_contents);
        intention_display = "Contents of: " + param_file;
    }
    
    if (param_intent == "X")
    {
        while (!interrupted)
        {
            std::cout << "Enter your Intention: ";
            if (!std::getline(std::cin, intention))
            {
                // If getline fails (e.g., due to an interrupt), break out of the loop immediately
                interrupted.store(true); // Ensure the flag is set if not already
                return 0;
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
        intention_value = intention;
    }
    else
    {
        intention = param_intent;
        intention_value = intention;
        intention_display = param_intent;
    }

    if (param_file != "X")
    {
        // Keep adding intention_value onto intention until its length is >= length of file_contents
        while (intention.length() < file_contents.length())
        {
            intention += intention_value;
        }
        intention += file_contents;
        intention_display += " (" + param_file + ")";
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
        // Convert useHashing to lowercase
        transform(useHashing.begin(), useHashing.end(), useHashing.begin(), ::tolower);
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
        // Convert useCompression to lowercase
        transform(useCompression.begin(), useCompression.end(), useCompression.begin(), ::tolower);
    }

    string intentionMultiplied, intentionHashed;
    size_t ramSize = 1024ULL * 1024 * 512 * numGBToUse;
    size_t multiplier = 0, hashMultiplier = 1;

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

    // Allocate memory on the GPU for intentionMultiplied and freq
    char *d_intentionMultiplied;
    unsigned long long int *d_freq;
    cudaMalloc(&d_intentionMultiplied, intentionMultiplied.size());
    cudaMalloc(&d_freq, sizeof(unsigned long long int));

    // Copy intentionMultiplied to the GPU
    cudaMemcpy(d_intentionMultiplied, intentionMultiplied.c_str(), intentionMultiplied.size(), cudaMemcpyHostToDevice);

    string totalIterations = "0", totalFreq = "0";
    unsigned long long freq = 0, seconds = 0;

    while (!interrupted)
    {
        auto start = high_resolution_clock::now();
        auto end = start + chrono::duration_cast<chrono::seconds>(chrono::seconds(1));

        // Set freq to 0 on the GPU
        cudaMemset(d_freq, 0, sizeof(unsigned long long int));

        while (high_resolution_clock::now() < end)
        {
            // Launch the CUDA kernel for intention repeating and frequency updating
            int blockSize = 256;
            int numBlocks = (intentionMultiplied.size() + blockSize - 1) / blockSize;
            intentionRepeaterKernel<<<numBlocks, blockSize>>>(d_intentionMultiplied, d_freq, intentionMultiplied.size());

            // Wait for the GPU to finish before accessing on host
            cudaDeviceSynchronize();

            // Copy the updated freq back to the CPU
            cudaMemcpy(&freq, d_freq, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        }

        totalFreq = MultiplyStrings(to_string(freq), to_string(multiplier));
        totalFreq = MultiplyStrings(totalFreq, to_string(hashMultiplier));
        totalIterations = FindSum(totalIterations, totalFreq);

        int digits = totalIterations.length();
        int freqDigits = totalFreq.length();
        ++seconds;
        freq = 0;

        std::cout << "[" + FormatTime(seconds) + "] "
                  << " (" << DisplaySuffix(totalIterations, digits - 1, "Iterations")
                  << " / " << DisplaySuffix(totalFreq, freqDigits - 1, "Frequency") << "Hz): " << intention_display
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

    std::cout << endl;
    // Free allocated memory on the GPU
    cudaFree(d_intentionMultiplied);
    cudaFree(d_freq);

    return 0;
}