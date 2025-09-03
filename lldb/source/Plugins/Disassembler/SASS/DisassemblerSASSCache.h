//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DISASSEMBLER_SASS_DISASSEMBLERSASSCACHE_H
#define LLDB_SOURCE_PLUGINS_DISASSEMBLER_SASS_DISASSEMBLERSASSCACHE_H

#include "lldb/Core/Address.h"
#include "lldb/Utility/FileSpec.h"
#include <llvm/Support/Error.h>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

/// Per-module SM architecture cache with its own mutex
/// This allows concurrent SM extraction for different modules
class ModuleSM {
public:
  explicit ModuleSM(const std::string &cache_key) : m_cache_key(cache_key) {}

  /// Thread-safe SM architecture extraction for this specific module
  llvm::Expected<std::string> FindSM(const lldb_private::Address &base_addr);

  llvm::Expected<std::string> FindSMForELFv7OrLower(lldb::ModuleSP module_sp);

  llvm::Expected<std::string> FindSMForELFv8OrGreater(lldb::ModuleSP module_sp);

private:
  std::string m_cache_key;
  std::optional<std::string> m_cached_sm_arch;
  std::mutex m_mutex;

  // Delete copy constructor and assignment operator
  ModuleSM(const ModuleSM &) = delete;
  ModuleSM &operator=(const ModuleSM &) = delete;
};

/// Singleton class to manage caches for DisassemblerSASS
/// This replaces static variables to follow LLVM best practices
class DisassemblerSASSCache {
public:
  static DisassemblerSASSCache &getInstance() {
    static DisassemblerSASSCache instance;
    return instance;
  }

  /// Find the nvdisasm executable (cached after first call)
  /// \return FileSpec for nvdisasm executable, or error if not found
  llvm::Expected<lldb_private::FileSpec> GetNvdisasmPath();

  // Per-module SM architecture caching
  std::shared_ptr<ModuleSM> GetModuleSM(const std::string &cache_key);

private:
  DisassemblerSASSCache() = default;
  ~DisassemblerSASSCache() = default;

  // Map of module cache keys to ModuleSM instances
  std::map<std::string, std::shared_ptr<ModuleSM>> m_module_sm_map;
  std::mutex m_module_map_mutex;

  // Delete copy constructor and assignment operator
  DisassemblerSASSCache(const DisassemblerSASSCache &) = delete;
  DisassemblerSASSCache &operator=(const DisassemblerSASSCache &) = delete;

  // nvdisasm path caching
  std::optional<lldb_private::FileSpec> m_cached_nvdisasm_path;
  std::once_flag m_search_once_nvdisasm;
};

#endif // LLDB_SOURCE_PLUGINS_DISASSEMBLER_SASS_DISASSEMBLERSASSCACHE_H
