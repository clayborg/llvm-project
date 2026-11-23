//===-- ModuleListTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ModuleList.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/UUID.h"

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

// Test that when we already have a module in the shared_module_list with a
// specific UUID, the next call to GetSharedModule with a module_spec with the
// same UUID should return the existing module instead of creating a new one.
TEST(ModuleListTest, GetSharedModuleReusesExistingModuleWithSameUUID) {
  SubsystemRAII<FileSystem, ObjectFileELF> subsystems;

  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x0000000000000010
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  // First, let's verify that calling GetSharedModule twice with the same
  // module_spec returns the same module pointer

  ModuleSP first_module;
  bool first_did_create = false;
  Status error_first =
      ModuleList::GetSharedModule(ExpectedFile->moduleSpec(), first_module,
                                  nullptr, &first_did_create, false);

  // Second call with the same spec
  ModuleSP second_module;
  bool second_did_create = false;
  Status error_second =
      ModuleList::GetSharedModule(ExpectedFile->moduleSpec(), second_module,
                                  nullptr, &second_did_create, false);

  if (error_first.Success() && error_second.Success()) {
    // If both succeeded, verify they're the same module
    EXPECT_EQ(first_module.get(), second_module.get())
        << "GetSharedModule should return the same module for the same spec";
    EXPECT_TRUE(first_did_create) << "First call should create the module";
    EXPECT_FALSE(second_did_create)
        << "Second call should reuse the existing module";
  }
}

// Test that UUID-based lookup finds existing modules
TEST(ModuleListTest, FindSharedModuleByUUID) {
  SubsystemRAII<FileSystem, ObjectFileELF> subsystems;

  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x0000000000000010
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  // Create and add a module to the shared module list using the moduleSpec()
  ModuleSP created_module;
  bool did_create = false;
  Status error = ModuleList::GetSharedModule(
      ExpectedFile->moduleSpec(), created_module, nullptr, &did_create, false);

  if (error.Success() && created_module) {
    // Get the UUID of the created module
    UUID module_uuid = created_module->GetUUID();

    if (module_uuid.IsValid()) {
      // Now try to find the module by UUID
      ModuleSP found_module = ModuleList::FindSharedModule(module_uuid);

      ASSERT_NE(found_module.get(), nullptr)
          << "FindSharedModule should find the module by UUID";
      EXPECT_EQ(found_module.get(), created_module.get())
          << "FindSharedModule should return the same module instance";
      EXPECT_EQ(found_module->GetUUID(), module_uuid)
          << "Found module should have the same UUID";
    }
  }
}

// Test that GetSharedModule with UUID finds existing module even with different
// path
TEST(ModuleListTest, GetSharedModuleByUUIDIgnoresPath) {
  SubsystemRAII<FileSystem, ObjectFileELF> subsystems;

  auto ExpectedFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x0000000000000010
...
)");
  ASSERT_THAT_EXPECTED(ExpectedFile, llvm::Succeeded());

  // Create and add a module to the shared module list
  ModuleSP first_module;
  bool first_did_create = false;
  Status first_error =
      ModuleList::GetSharedModule(ExpectedFile->moduleSpec(), first_module,
                                  nullptr, &first_did_create, false);

  if (first_error.Success() && first_module) {
    UUID module_uuid = first_module->GetUUID();

    if (module_uuid.IsValid()) {
      // Now try to get a module with the same UUID but different path
      ModuleSpec second_spec;
      second_spec.GetFileSpec() = FileSpec("/different/path/to/module.so");
      second_spec.GetArchitecture() = ArchSpec("x86_64-pc-linux");
      second_spec.GetUUID() = module_uuid;

      ModuleSP second_module;
      bool second_did_create = false;
      Status second_error = ModuleList::GetSharedModule(
          second_spec, second_module, nullptr, &second_did_create, false);

      if (second_error.Success() && second_module) {
        // If we got a module back, check if it's the same one
        bool is_same_module = (second_module.get() == first_module.get());

        // Document the behavior: ideally UUID should take precedence
        // and return the existing module
        EXPECT_TRUE(is_same_module)
            << "GetSharedModule with matching UUID should return existing "
               "module, "
            << "even with different path (per PR #160199)";

        if (is_same_module) {
          EXPECT_FALSE(second_did_create)
              << "Should not create a new module when UUID matches";
        }
      }
    }
  }
}

TEST(ModuleListTest, ReplaceEquivalentKeepsDifferentObjectSlices) {
  SubsystemRAII<FileSystem, ObjectFileELF> subsystems;

  auto DummyFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x0000000000000010
...
)");
  ASSERT_THAT_EXPECTED(DummyFile, llvm::Succeeded());

  auto SliceFile = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x0000000000000010
...
)");
  ASSERT_THAT_EXPECTED(SliceFile, llvm::Succeeded());

  ModuleSP dummy_module;
  bool did_create = false;
  Status error = ModuleList::GetSharedModule(
      DummyFile->moduleSpec(), dummy_module, nullptr, &did_create, false);
  ASSERT_TRUE(error.Success());
  ASSERT_NE(dummy_module, nullptr);
  ASSERT_NE(dummy_module->GetObjectFile(), nullptr);

  ModuleSpec slice_spec = SliceFile->moduleSpec();
  const llvm::sys::TimePoint<> object_mod_time;
  ModuleSP first_module = std::make_shared<Module>(
      slice_spec.GetFileSpec(), slice_spec.GetArchitecture(),
      ConstString("libwithslices.so[0x1000-0x1200)"), 0x1000, object_mod_time);
  ModuleSP second_module = std::make_shared<Module>(
      slice_spec.GetFileSpec(), slice_spec.GetArchitecture(),
      ConstString("libwithslices.so[0x2000-0x2300)"), 0x2000, object_mod_time);

  ASSERT_NE(first_module->GetObjectName(), second_module->GetObjectName());
  ASSERT_NE(first_module->GetObjectOffset(), second_module->GetObjectOffset());

  ModuleList module_list;
  module_list.Append(dummy_module);
  module_list.Append(first_module);
  module_list.ReplaceEquivalent(second_module);

  EXPECT_EQ(module_list.GetSize(), 3u);
  EXPECT_EQ(module_list.GetModuleAtIndex(0), dummy_module);
  EXPECT_EQ(module_list.GetModuleAtIndex(1), first_module);
  EXPECT_EQ(module_list.GetModuleAtIndex(2), second_module);
}

static constexpr llvm::StringLiteral kSliceYaml = R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x0000000000000010
...
)";

/// The offsets of the two ELF images in the file WriteTwoSliceFile writes.
/// Neither is at offset 0: a spec at offset 0 that covers its whole data
/// buffer selects no slice.
static constexpr uint64_t kFirstSliceOffset = 0x1000;
static constexpr uint64_t kSecondSliceOffset = 0x2000;

/// Write two copies of \p image into one temporary file, at kFirstSliceOffset
/// and kSecondSliceOffset, and set \p first_spec and \p second_spec to the
/// specs that select each copy, the way ProcessNVGPUCore selects the cubins
/// embedded in a GPU core file.
static llvm::Expected<llvm::sys::fs::TempFile>
WriteTwoSliceFile(TestFile &image, ModuleSpec &first_spec,
                  ModuleSpec &second_spec) {
  llvm::Expected<llvm::sys::fs::TempFile> temp =
      llvm::sys::fs::TempFile::create("slices%%%%%%%%%%%%%%%%");
  if (!temp)
    return temp.takeError();
  DataBufferSP data = image.moduleSpec().GetData();
  const uint64_t size = data->GetByteSize();
  {
    llvm::raw_fd_ostream os(temp->FD, /*shouldClose=*/false);
    os.write_zeros(kFirstSliceOffset);
    os.write(reinterpret_cast<const char *>(data->GetBytes()), size);
    os.write_zeros(kSecondSliceOffset - kFirstSliceOffset - size);
    os.write(reinterpret_cast<const char *>(data->GetBytes()), size);
  }
  FileSpec file(temp->TmpName);
  first_spec = ModuleSpec(file, UUID(), data);
  first_spec.SetObjectOffset(kFirstSliceOffset);
  first_spec.SetObjectSize(size);
  second_spec = ModuleSpec(file, UUID(), data);
  second_spec.SetObjectOffset(kSecondSliceOffset);
  second_spec.SetObjectSize(size);
  return std::move(*temp);
}

// Slices of one file are separate shared modules, and a lookup of either one
// returns its own module.
TEST(ModuleListTest, GetSharedModuleKeepsSlicesOfOneFileApart) {
  SubsystemRAII<FileSystem, ObjectFileELF> subsystems;
  auto image = TestFile::fromYaml(kSliceYaml);
  ASSERT_THAT_EXPECTED(image, llvm::Succeeded());
  ModuleSpec first_spec, second_spec;
  auto file = WriteTwoSliceFile(*image, first_spec, second_spec);
  ASSERT_THAT_EXPECTED(file, llvm::Succeeded());

  ModuleSP first_module, second_module, module;
  bool did_create = false;
  Status error = ModuleList::GetSharedModule(first_spec, first_module, nullptr,
                                             &did_create);
  ASSERT_TRUE(error.Success()) << error.AsCString();
  EXPECT_TRUE(did_create);
  ASSERT_TRUE(ModuleList::GetSharedModule(second_spec, second_module, nullptr,
                                          &did_create)
                  .Success());
  EXPECT_TRUE(did_create);
  ASSERT_NE(first_module, nullptr);
  ASSERT_NE(second_module, nullptr);
  EXPECT_NE(first_module, second_module);
  EXPECT_EQ(second_module->GetObjectOffset(), kSecondSliceOffset);

  // A second lookup of each slice reuses its module.
  ASSERT_TRUE(
      ModuleList::GetSharedModule(first_spec, module, nullptr, &did_create)
          .Success());
  EXPECT_FALSE(did_create);
  EXPECT_EQ(module, first_module);
  ASSERT_TRUE(
      ModuleList::GetSharedModule(second_spec, module, nullptr, &did_create)
          .Success());
  EXPECT_FALSE(did_create);
  EXPECT_EQ(module, second_module);

  ModuleList found;
  ModuleList::FindSharedModules(second_spec, found);
  ASSERT_EQ(found.GetSize(), 1u);
  EXPECT_EQ(found.GetModuleAtIndex(0), second_module);

  // A slice with no module yet matches nothing.
  ModuleSpec missing_spec = second_spec;
  missing_spec.SetObjectOffset(kSecondSliceOffset + kFirstSliceOffset);
  found.Clear();
  ModuleList::FindSharedModules(missing_spec, found);
  EXPECT_EQ(found.GetSize(), 0u);

  llvm::consumeError(file->discard());
}

// Creating a slice again replaces the older copy of that slice, and only that
// one, and reports it as an old module. Once released, both slices are
// removed as orphans.
TEST(ModuleListTest, GetSharedModuleReplacesOlderCopyOfSlice) {
  SubsystemRAII<FileSystem, ObjectFileELF> subsystems;
  auto image = TestFile::fromYaml(kSliceYaml);
  ASSERT_THAT_EXPECTED(image, llvm::Succeeded());
  ModuleSpec first_spec, second_spec;
  auto file = WriteTwoSliceFile(*image, first_spec, second_spec);
  ASSERT_THAT_EXPECTED(file, llvm::Succeeded());

  ModuleSP first_module, old_second_module, new_second_module;
  ASSERT_TRUE(
      ModuleList::GetSharedModule(first_spec, first_module, nullptr, nullptr)
          .Success());
  ASSERT_TRUE(ModuleList::GetSharedModule(second_spec, old_second_module,
                                          nullptr, nullptr)
                  .Success());

  llvm::SmallVector<ModuleSP, 1> old_modules;
  bool did_create = false;
  ASSERT_TRUE(ModuleList::GetSharedModule(second_spec, new_second_module,
                                          &old_modules, &did_create,
                                          /*always_create=*/true)
                  .Success());
  EXPECT_TRUE(did_create);
  ASSERT_NE(new_second_module, nullptr);
  EXPECT_NE(new_second_module, old_second_module);
  ASSERT_EQ(old_modules.size(), 1u);
  EXPECT_EQ(old_modules[0], old_second_module);

  ModuleList found;
  ModuleList::FindSharedModules(second_spec, found);
  ASSERT_EQ(found.GetSize(), 1u);
  EXPECT_EQ(found.GetModuleAtIndex(0), new_second_module);
  found.Clear();
  ModuleList::FindSharedModules(first_spec, found);
  ASSERT_EQ(found.GetSize(), 1u);
  EXPECT_EQ(found.GetModuleAtIndex(0), first_module);
  found.Clear();

  old_modules.clear();
  first_module.reset();
  old_second_module.reset();
  new_second_module.reset();
  ModuleList::RemoveOrphanSharedModules(/*mandatory=*/true);
  ModuleList::FindSharedModules(first_spec, found);
  ModuleList::FindSharedModules(second_spec, found);
  EXPECT_EQ(found.GetSize(), 0u);

  llvm::consumeError(file->discard());
}
