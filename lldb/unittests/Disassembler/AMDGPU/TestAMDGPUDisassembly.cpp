//===-- TestAMDGPUDisassembly.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/ArchSpec.h"

#include "Plugins/Disassembler/LLVMC/DisassemblerLLVMC.h"

using namespace lldb;
using namespace lldb_private;

namespace {
class TestAMDGPUDisassembly : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();
};

void TestAMDGPUDisassembly::SetUpTestCase() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  DisassemblerLLVMC::Initialize();
}

void TestAMDGPUDisassembly::TearDownTestCase() {
  DisassemblerLLVMC::Terminate();
}
} // namespace

// AMDGPU's symbolizer reads the disassembler's DisInfo as a symbol list, so
// decoding a branch with LLDB's disassembler as DisInfo read that object as a
// vector and crashed.
TEST_F(TestAMDGPUDisassembly, TestBranch) {
  ArchSpec arch("amdgcn-amd-amdhsa");

  const unsigned num_of_instructions = 2;
  uint8_t data[] = {
      0xf0, 0xff, 0x82, 0xbf, // s_branch 65520
      0x00, 0x00, 0x81, 0xbf, // s_endpgm
  };

  Address start_addr(0x100);
  DisassemblerSP disass_sp = Disassembler::DisassembleBytes(
      arch, nullptr, nullptr, "gfx942", nullptr, start_addr, &data,
      sizeof(data), num_of_instructions, false);
  ASSERT_TRUE(disass_sp);

  const InstructionList inst_list(disass_sp->GetInstructionList());
  ASSERT_EQ(num_of_instructions, inst_list.GetSize());

  ExecutionContext exe_ctx;
  InstructionSP inst_sp = inst_list.GetInstructionAtIndex(0);
  EXPECT_TRUE(inst_sp->DoesBranch());
  EXPECT_STREQ("s_branch", inst_sp->GetMnemonic(&exe_ctx));
  EXPECT_STREQ("65520", inst_sp->GetOperands(&exe_ctx));

  inst_sp = inst_list.GetInstructionAtIndex(1);
  EXPECT_STREQ("s_endpgm", inst_sp->GetMnemonic(&exe_ctx));
}
