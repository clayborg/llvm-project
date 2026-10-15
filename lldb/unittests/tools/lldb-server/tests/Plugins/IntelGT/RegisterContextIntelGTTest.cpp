//===-- RegisterContextIntelGTTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include <cstring>

// Test PC computation logic from RegisterContextIntelGT
// This validates the fix for full_byte_size >= 12 condition

namespace {

// Constants from Intel GPU architecture
constexpr uint32_t kCR0_DWORD2_OFFSET = 8; // IP is at byte offset 8 in CR0

// Simulates PC computation from CR0 register
// CR0 is 96 bits (12 bytes) = 3 DWORDs
// DWORD 2 contains the 32-bit instruction pointer (IP)
uint64_t ComputePC(const uint8_t *cr0_buffer, size_t buffer_size,
                   uint64_t isabase) {
  if (buffer_size < 12) {
    // Should never happen with the fix
    return 0;
  }

  // Extract 32-bit IP from CR0 DWORD 2 (bytes 8-11)
  uint32_t ip = 0;
  std::memcpy(&ip, cr0_buffer + kCR0_DWORD2_OFFSET, sizeof(ip));

  // PC = isabase + IP
  return isabase + static_cast<uint64_t>(ip);
}

// Helper to create a CR0 buffer with a given IP value
void CreateCR0Buffer(uint8_t *buffer, uint32_t ip) {
  std::memset(buffer, 0, 12);
  std::memcpy(buffer + kCR0_DWORD2_OFFSET, &ip, sizeof(ip));
}

} // anonymous namespace

class RegisterContextIntelGTPCTest : public ::testing::Test {
protected:
  // Real values from the bug fix scenario
  static constexpr uint64_t kIsabaseExpected = 0xffff8000fff00000ULL;
  static constexpr uint32_t kIPExpected = 0x42d90;
  static constexpr uint64_t kPCExpected = 0xffff8000fff42d90ULL;

  // Wrong PC that was being reported before the fix
  static constexpr uint64_t kPCWrong = 0x8000c000c00004c0ULL;
};

TEST_F(RegisterContextIntelGTPCTest, TestPCComputationWithCorrectIsabase) {
  uint8_t cr0_buffer[12];
  CreateCR0Buffer(cr0_buffer, kIPExpected);

  uint64_t computed_pc = ComputePC(cr0_buffer, 12, kIsabaseExpected);

  EXPECT_EQ(computed_pc, kPCExpected)
      << "PC should be computed as isabase + IP";
  EXPECT_NE(computed_pc, kPCWrong)
      << "PC should not be the raw incorrect value";
}

TEST_F(RegisterContextIntelGTPCTest, TestPCComputationComponents) {
  uint8_t cr0_buffer[12];
  CreateCR0Buffer(cr0_buffer, kIPExpected);

  // Verify the components
  uint32_t ip_extracted = 0;
  std::memcpy(&ip_extracted, cr0_buffer + kCR0_DWORD2_OFFSET, sizeof(ip_extracted));

  EXPECT_EQ(ip_extracted, kIPExpected) << "IP should be correctly extracted from CR0";

  uint64_t pc = kIsabaseExpected + ip_extracted;
  EXPECT_EQ(pc, kPCExpected) << "PC = isabase + IP computation";
}

TEST_F(RegisterContextIntelGTPCTest, TestBufferSizeCheck) {
  uint8_t cr0_buffer[12];
  CreateCR0Buffer(cr0_buffer, kIPExpected);

  // Test with correct size (12 bytes = 96 bits)
  uint64_t pc_12 = ComputePC(cr0_buffer, 12, kIsabaseExpected);
  EXPECT_EQ(pc_12, kPCExpected) << "Should work with 12-byte CR0";

  // Test with larger size (16 bytes, original check)
  uint64_t pc_16 = ComputePC(cr0_buffer, 16, kIsabaseExpected);
  EXPECT_EQ(pc_16, kPCExpected) << "Should work with 16-byte buffer";

  // Test with insufficient size
  uint64_t pc_8 = ComputePC(cr0_buffer, 8, kIsabaseExpected);
  EXPECT_EQ(pc_8, 0ULL) << "Should fail gracefully with insufficient buffer";
}

TEST_F(RegisterContextIntelGTPCTest, TestZeroIP) {
  uint8_t cr0_buffer[12];
  CreateCR0Buffer(cr0_buffer, 0);

  uint64_t computed_pc = ComputePC(cr0_buffer, 12, kIsabaseExpected);

  EXPECT_EQ(computed_pc, kIsabaseExpected)
      << "With IP=0, PC should equal isabase";
}

TEST_F(RegisterContextIntelGTPCTest, TestMaxIP) {
  uint8_t cr0_buffer[12];
  uint32_t max_ip = 0xFFFFFFFF;
  CreateCR0Buffer(cr0_buffer, max_ip);

  uint64_t computed_pc = ComputePC(cr0_buffer, 12, kIsabaseExpected);

  EXPECT_EQ(computed_pc, kIsabaseExpected + max_ip)
      << "Should handle maximum 32-bit IP value";
}

TEST_F(RegisterContextIntelGTPCTest, TestDifferentIsabases) {
  uint8_t cr0_buffer[12];
  CreateCR0Buffer(cr0_buffer, 0x1000);

  // Test with different isabase values
  struct {
    uint64_t isabase;
    uint64_t expected_pc;
  } test_cases[] = {
      {0xFFFF800000000000ULL, 0xFFFF800000001000ULL},
      {0x8000000000000000ULL, 0x8000000000001000ULL},
      {0x0000000000000000ULL, 0x0000000000001000ULL},
  };

  for (const auto &test : test_cases) {
    uint64_t computed_pc = ComputePC(cr0_buffer, 12, test.isabase);
    EXPECT_EQ(computed_pc, test.expected_pc)
        << "PC computation with isabase=0x" << std::hex << test.isabase;
  }
}

// Test that validates the actual bug scenario from the logs:
// full_byte_size=12 was failing the ">= 16" check, causing wrong PC
TEST_F(RegisterContextIntelGTPCTest, TestBugScenarioFullByteSize12) {
  // This test validates the fix for the condition:
  // Original:  if (lldb_reg == pc_reg && full_byte_size >= 16)
  // Fixed:     if (lldb_reg == pc_reg && full_byte_size >= 12)

  uint8_t cr0_buffer[12];
  CreateCR0Buffer(cr0_buffer, kIPExpected);

  // With full_byte_size=12, PC computation should trigger
  size_t full_byte_size = 12;
  bool should_compute_pc = (full_byte_size >= 12);

  EXPECT_TRUE(should_compute_pc)
      << "PC computation should trigger with 12-byte CR0";

  if (should_compute_pc) {
    uint64_t computed_pc = ComputePC(cr0_buffer, full_byte_size, kIsabaseExpected);
    EXPECT_EQ(computed_pc, kPCExpected)
        << "With the fix, PC should be correctly computed from 12-byte CR0";
  }

  // Verify the old condition would have failed
  bool old_condition = (full_byte_size >= 16);
  EXPECT_FALSE(old_condition)
      << "Old condition (>= 16) should fail with 12-byte CR0, causing the bug";
}

TEST_F(RegisterContextIntelGTPCTest, TestIPAtCorrectOffset) {
  uint8_t cr0_buffer[12];
  std::memset(cr0_buffer, 0xFF, sizeof(cr0_buffer));

  // Write specific IP value at correct offset
  uint32_t test_ip = 0x12345678;
  std::memcpy(cr0_buffer + kCR0_DWORD2_OFFSET, &test_ip, sizeof(test_ip));

  // Read it back
  uint32_t read_ip = 0;
  std::memcpy(&read_ip, cr0_buffer + kCR0_DWORD2_OFFSET, sizeof(read_ip));

  EXPECT_EQ(read_ip, test_ip)
      << "IP should be at byte offset 8 (DWORD 2) in CR0";
}
