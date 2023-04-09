extern "C" {
#include <asm/atomic.h>
}
#include "thread.h"

//#include "array.hpp"
//#include "deref_scope.hpp
//#include "device.hpp"
#include "local_concurrent_hopscotch.hpp"
#include "helpers.hpp"
//#include "manager.hpp"
#include "snappy.h"
//#include "stats.hpp"
#include "zipf.hpp"

// crypto++
#include "cryptopp/aes.h"
#include "cryptopp/filters.h"
#include "cryptopp/modes.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include <fstream>
#include <numeric>

#include "pgfault.h"

//using namespace far_memory;

// #define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))

//namespace far_memory {
class FarMemTest {
private:
  // Hashtable.
  constexpr static uint32_t kKeyLen = 12;
  constexpr static uint32_t kValueLen = 4;
  constexpr static uint32_t kLocalHashTableNumEntriesShift = 28;
  // constexpr static uint32_t kRemoteHashTableNumEntriesShift = 28;
  constexpr static uint64_t kRemoteHashTableSlabSize = (4ULL << 30) * 1.05;
  constexpr static uint32_t kNumKVPairs = (1 << 27);

  // Array.
  constexpr static uint32_t kNumArrayEntries = (2<<20);
  constexpr static uint32_t kArrayEntrySize = 8192;     // 8 K

  // Runtime.
  constexpr static uint32_t kNumMutatorThreads = 40;
  constexpr static double kZipfParamS = 0.85;
  constexpr static uint32_t kNumKeysPerRequest = 32;
  constexpr static uint32_t kNumReqs = kNumKVPairs / kNumKeysPerRequest;
  constexpr static uint32_t kLog10NumKeysPerRequest =
      helpers::static_log(10, kNumKeysPerRequest);
  constexpr static uint32_t kReqLen = kKeyLen - kLog10NumKeysPerRequest;
  constexpr static uint32_t kReqSeqLen = kNumReqs;

  // Output.
  constexpr static uint32_t kPrintPerIters = 8192;
  constexpr static uint32_t kMaxPrintIntervalUs = 1000 * 1000; // 1 second(s).
  constexpr static uint32_t kPrintTimes = 100;

  struct Req {
    char data[kReqLen];
  };

  struct Key {
    char data[kKeyLen];
  };

  union Value {
    uint32_t num;
    char data[kValueLen];
  };

  struct ArrayEntry {
    uint8_t data[kArrayEntrySize];
  };

  struct alignas(64) Cnt {
    uint64_t c;
  };

//  using AppArray = Array<ArrayEntry, kNumArrayEntries>;

  std::unique_ptr<std::mt19937> generators[helpers::kNumCPUs];
  alignas(helpers::kHugepageSize) Req all_gen_reqs[kNumReqs];
  uint32_t all_zipf_req_indices[helpers::kNumCPUs][kReqSeqLen];
  
  Cnt req_cnts[kNumMutatorThreads];
//  Cnt local_array_miss_cnts[kNumMutatorThreads];
//  Cnt local_hashtable_miss_cnts[kNumMutatorThreads];
  Cnt per_core_req_idx[helpers::kNumCPUs];

  std::atomic_flag flag;
  uint64_t print_times = 0;
  uint64_t prev_sum_reqs = 0;
  uint64_t prev_sum_array_misses = 0;
  uint64_t prev_sum_hashtable_misses = 0;
  uint64_t prev_us = 0;
  uint64_t running_us = 0;
  std::vector<double> mops_records;
//  std::vector<double> hashtable_miss_rate_records;
//  std::vector<double> array_miss_rate_records;

  unsigned char key[CryptoPP::AES::DEFAULT_KEYLENGTH];
  unsigned char iv[CryptoPP::AES::BLOCKSIZE];
  std::unique_ptr<CryptoPP::CBC_Mode_ExternalCipher::Encryption> cbcEncryption;
  std::unique_ptr<CryptoPP::AES::Encryption> aesEncryption;

  inline void append_uint32_to_char_array(uint32_t n, uint32_t suffix_len,
                                          char *array) {
    uint32_t len = 0;
    while (n) {
      auto digit = n % 10;
      array[len++] = digit + '0';
      n = n / 10;
    }
    while (len < suffix_len) {
      array[len++] = '0';
    }
    std::reverse(array, array + suffix_len);
  }

  inline void random_string(char *data, uint32_t len) {
    BUG_ON(len <= 0);
    preempt_disable();
    auto guard = helpers::finally([&]() { preempt_enable(); });
    auto &generator = *generators[ get_core_num_v2()];
    std::uniform_int_distribution<int> distribution('a', 'z' + 1);
    for (uint32_t i = 0; i < len; i++) {
      data[i] = char(distribution(generator));
    }
  }

  inline void random_req(char *data, uint32_t tid) {
    auto tid_len = helpers::static_log(10, kNumMutatorThreads);
    random_string(data, kReqLen - tid_len);
    append_uint32_to_char_array(tid, tid_len, data + kReqLen - tid_len);
  }

  inline uint32_t random_uint32() {
    preempt_disable();
    auto guard = helpers::finally([&]() { preempt_enable(); });
    auto &generator = *generators[ get_core_num_v2()];
    std::uniform_int_distribution<uint32_t> distribution(
        0, std::numeric_limits<uint32_t>::max());
    return distribution(generator);
  }

  /* calculate zipf curve */
  void calculate_and_dump_zipf_curves() {
    std::cout << "dumping zipf curves" << std::endl;
    uint32_t buckets = 1000;
    std::vector<double> zipf_curve(kNumReqs);
    for (uint32_t c = 0; c < helpers::kNumCPUs; c++) {
      std::fill(zipf_curve.begin(), zipf_curve.end(), 0);
      for (uint32_t i = 0; i < kReqSeqLen; i++)
        zipf_curve[all_zipf_req_indices[c][i]]++;
      std::sort(zipf_curve.begin(), zipf_curve.end(), std::greater<uint32_t>());
      // collapse into 1000 buckets
      std::vector<double> zipf_curve_collapsed(buckets);
      for (uint32_t i = 0; i < buckets; i++) {
        uint32_t start = i * (kNumReqs / buckets);
        uint32_t end = (i + 1) * (kNumReqs / buckets);
        for (uint32_t j = start; j < end; j++) {
          zipf_curve_collapsed[i] += zipf_curve[j];
        }
      }
      auto sum = std::accumulate(zipf_curve.begin(), zipf_curve.end(), 0);
      for (auto &x : zipf_curve_collapsed)
        x /= sum;
      std::ofstream ofs("zipf_curve_" + std::to_string(c) + ".txt");
      for (auto &x : zipf_curve_collapsed)
        ofs << x << std::endl;
    }
  }

  void prepare(LocalGenericConcurrentHopscotch *hopscotch) {
    for (uint32_t i = 0; i < helpers::kNumCPUs; i++) {
      std::random_device rd;
      generators[i].reset(new std::mt19937(rd()));
    }
    memset(key, 0x00, CryptoPP::AES::DEFAULT_KEYLENGTH);
    memset(iv, 0x00, CryptoPP::AES::BLOCKSIZE);
    aesEncryption.reset(
        new CryptoPP::AES::Encryption(key, CryptoPP::AES::DEFAULT_KEYLENGTH));
    cbcEncryption.reset(
        new CryptoPP::CBC_Mode_ExternalCipher::Encryption(*aesEncryption, iv));
    std::vector<rt::Thread> threads;
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back(rt::Thread([&, tid]() {
        auto num_reqs_per_thread = kNumReqs / kNumMutatorThreads;
        auto req_offset = tid * num_reqs_per_thread;
        auto *thread_gen_reqs = &all_gen_reqs[req_offset];
        for (uint32_t i = 0; i < num_reqs_per_thread; i++) {
          Req req;
          random_req(req.data, tid);
          Key key;
          memcpy(key.data, req.data, kReqLen);
          for (uint32_t j = 0; j < kNumKeysPerRequest; j++) {
            append_uint32_to_char_array(j, kLog10NumKeysPerRequest,
                                        key.data + kReqLen);
            Value value;
            value.num = (j ? 0 : req_offset + i);
            hopscotch->put(kKeyLen, (const uint8_t *)key.data, kValueLen,
                           (uint8_t *)value.data);
          }
          thread_gen_reqs[i] = req;
        }
      }));
    }
    for (auto &thread : threads) {
      thread.Join();
    }

    std::cout << "generating zipf distribution" << std::endl;
    preempt_disable();
    zipf_table_distribution<> zipf(kNumReqs, kZipfParamS);
    auto &generator = generators[ get_core_num_v2()];
    constexpr uint32_t kPerCoreWinInterval = kReqSeqLen / helpers::kNumCPUs;
    for (uint32_t i = 0; i < kReqSeqLen; i++) {
      auto rand_idx = zipf(*generator);
      for (uint32_t j = 0; j < helpers::kNumCPUs; j++) {
        all_zipf_req_indices[j][(i + (j * kPerCoreWinInterval)) % kReqSeqLen] =
            rand_idx;
      }
    }
    preempt_enable();

    // calculate_and_dump_zipf_curves();
  }

//  void prepare(AppArray *array) {
    // We may put something into array for initialization.
    // But for the performance benchmark, we just do nothing here.
//  }

  void consume_array_entry(const ArrayEntry &entry) {
    std::string ciphertext;
    CryptoPP::StreamTransformationFilter stfEncryptor(
        *cbcEncryption, new CryptoPP::StringSink(ciphertext));
    stfEncryptor.Put((const unsigned char *)&entry.data, sizeof(entry));
    stfEncryptor.MessageEnd();
    std::string compressed;
    snappy::Compress(ciphertext.c_str(), ciphertext.size(), &compressed);
    auto compressed_len = compressed.size();
    ACCESS_ONCE(compressed_len);
  }

  void print_perf() {
    if (!flag.test_and_set()) {
      preempt_disable();
      auto us = microtime();
      uint64_t sum_reqs = 0;
//      uint64_t sum_hashtable_misses = 0;
//      uint64_t sum_array_misses = 0;
      for (uint32_t i = 0; i < kNumMutatorThreads; i++) {
        sum_reqs += ACCESS_ONCE(req_cnts[i].c);
//        sum_hashtable_misses += ACCESS_ONCE(local_hashtable_miss_cnts[i].c);
//        sum_array_misses += ACCESS_ONCE(local_array_miss_cnts[i].c);
      }
      if (us - prev_us > kMaxPrintIntervalUs) {
        auto mops =
            ((double)(sum_reqs - prev_sum_reqs) / (us - prev_us)) * 1.098;
//        auto hashtable_miss_rate =
//            (double)(sum_hashtable_misses - prev_sum_hashtable_misses) /
//            (kNumKeysPerRequest * (sum_reqs - prev_sum_reqs));
//        auto array_miss_rate =
//            (double)(sum_array_misses - prev_sum_array_misses) /
//            (sum_reqs - prev_sum_reqs);
        mops_records.push_back(mops);
//        hashtable_miss_rate_records.push_back(hashtable_miss_rate);
//        array_miss_rate_records.push_back(array_miss_rate);
        us = microtime();
        running_us += (us - prev_us);
        if (print_times++ >= kPrintTimes) {
          constexpr double kRatioChosenRecords = 0.1;
          uint32_t num_chosen_records =
              mops_records.size() * kRatioChosenRecords;
          mops_records.erase(mops_records.begin(),
                             mops_records.end() - num_chosen_records);
//          hashtable_miss_rate_records.erase(hashtable_miss_rate_records.begin(),
//                                            hashtable_miss_rate_records.end() -
//                                                num_chosen_records);
//          array_miss_rate_records.erase(array_miss_rate_records.begin(),
//                                        array_miss_rate_records.end() -
//                                            num_chosen_records);
          std::cout << "mops = "
                    << accumulate(mops_records.begin(), mops_records.end(),
                                  0.0) /
                           mops_records.size()
                    << std::endl;
//          std::cout << "hashtable miss rate = "
//                    << accumulate(hashtable_miss_rate_records.begin(),
//                                  hashtable_miss_rate_records.end(), 0.0) /
//                           hashtable_miss_rate_records.size()
//                    << std::endl;
//          std::cout << "array miss rate = "
//                    << accumulate(array_miss_rate_records.begin(),
//                                  array_miss_rate_records.end(), 0.0) /
//                           array_miss_rate_records.size()
//                    << std::endl;
    	    save_checkpoint("run_end");
          exit(0);
        }
        prev_us = us;
        prev_sum_reqs = sum_reqs;
//        prev_sum_array_misses = sum_array_misses;
//        prev_sum_hashtable_misses = sum_hashtable_misses;
      }
      preempt_enable();
      flag.clear();
    }
  }

  void bench(LocalGenericConcurrentHopscotch *hopscotch, ArrayEntry *array) {
    std::vector<rt::Thread> threads;
    prev_us = microtime();
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back(rt::Thread([&, tid]() {
        uint32_t cnt = 0;
        printf("benching on core: %d\n", get_core_num_v2());
        while (1) {
          if (unlikely(cnt++ % kPrintPerIters == 0)) {
            preempt_disable();
            print_perf();
            preempt_enable();
          }
          preempt_disable();
          auto core_num =  get_core_num_v2();
          auto req_idx =
              all_zipf_req_indices[core_num][per_core_req_idx[core_num].c];
          if (unlikely(++per_core_req_idx[core_num].c == kReqSeqLen)) {
            per_core_req_idx[core_num].c = 0;
          }
          preempt_enable();

          auto &req = all_gen_reqs[req_idx];
          Key key;
          memcpy(key.data, req.data, kReqLen);
          uint32_t array_index = 0;
          {
            for (uint32_t i = 0; i < kNumKeysPerRequest; i++) {
              append_uint32_to_char_array(i, kLog10NumKeysPerRequest,
                                          key.data + kReqLen);
              Value value;
              uint16_t value_len;
//              bool forwarded = false;
//              hopscotch->_get(kKeyLen, (const uint8_t *)key.data,
//                              &value_len, (uint8_t *)value.data, &forwarded);
                hopscotch->get(kKeyLen, (const uint8_t *)key.data,
                              &value_len, (uint8_t *)value.data, false);
//              ACCESS_ONCE(local_hashtable_miss_cnts[tid].c) += forwarded;
              array_index += value.num;
            }
          }
          {
            array_index %= kNumArrayEntries;
            const auto &array_entry = array[array_index];
#ifdef SET_PRIORITY
            hint_read_fault_prio((void*) array_entry.data, 1);
            hint_read_fault_prio((void*) ((unsigned long)(array_entry.data) + EDEN_PAGE_SIZE), 1);
#else
            hint_read_fault((void*) array_entry.data);
            hint_read_fault((void*) ((unsigned long)(array_entry.data) + EDEN_PAGE_SIZE));
#endif
            preempt_disable();
            consume_array_entry(array_entry);
            preempt_enable();
          }
          preempt_disable();
          core_num =  get_core_num_v2();
          preempt_enable();
          ACCESS_ONCE(req_cnts[tid].c)++;
        }
      }));
    }
    for (auto &thread : threads) {
      thread.Join();
    }
  }

public:
  void do_work() {
    auto hopscotch = new LocalGenericConcurrentHopscotch(kLocalHashTableNumEntriesShift, kRemoteHashTableSlabSize);
    
    std::cout << "Prepare..." << std::endl;
    save_checkpoint("preload_start");
    prepare(hopscotch);
    save_checkpoint("preload_end");

//    auto array_ptr = std::unique_ptr<AppArray>(
//        manager->allocate_array_heap<ArrayEntry, kNumArrayEntries>());
#ifdef INITIALIZE_ARRAY
    /* allocate and touch the memory; used to find out max memory */
    auto array_ptr = new ArrayEntry[kNumArrayEntries+1]();
#else
    /* just allocate memory; this is the default */
    auto array_ptr = new ArrayEntry[kNumArrayEntries+1];
#endif

    /* making sure the entries aligns with pages */
    array_ptr = (FarMemTest::ArrayEntry*) align_up((unsigned long) array_ptr, EDEN_PAGE_SIZE);
//    array_ptr->disable_prefetch();
//    prepare(array_ptr);
    
    std::cout << "Bench..." << std::endl;
    save_checkpoint("run_start");
    bench(hopscotch, array_ptr);
  }

  void run() {
    BUG_ON(madvise(all_gen_reqs, sizeof(Req) * kNumReqs, MADV_HUGEPAGE) != 0);
//    std::unique_ptr<FarMemManager> manager =
//        std::unique_ptr<FarMemManager>(FarMemManagerFactory::build(
//            kCacheSize, kNumGCThreads,
//            new TCPDevice(raddr, kNumConnections, kFarMemSize)));
    do_work();
  }

  ~FarMemTest() {
    /* do nothing */
  }
};
//} // namespace far_memory

int argc;
FarMemTest test;
void _main(void *arg) {
  /* write pid and wait some time for the saved pid to be added to 
   * the cgroup to enforce fastswap limits */
	std::cout << "writing out pid " << getpid() << std::endl;
	fwrite_number("main_pid", getpid());
  sleep(1);

  // char **argv = (char **)arg;
  // std::string ip_addr_port(argv[1]);
  // auto raddr = helpers::str_to_netaddr(ip_addr_port);
  test.run();
}

int main(int _argc, char *argv[]) {
  int ret;

  if (_argc < 2) {
    std::cerr << "usage: [cfg_file]" << std::endl;
    return -EINVAL;
  }

  char conf_path[strlen(argv[1]) + 1];
  strcpy(conf_path, argv[1]);
  for (int i = 2; i < _argc; i++) {
    argv[i - 1] = argv[i];
  }
  argc = _argc - 1;

  ret = runtime_init(conf_path, _main, argv);
  if (ret) {
    std::cerr << "failed to start runtime" << std::endl;
    return ret;
  }

  return 0;
}
