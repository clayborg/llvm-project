//===- TrieStrtab.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TrieBuilder is a class that will build a string table in Trie format that
// is designed to replace the standard string tables from DWARF and object
// files. It allows 32 bit offsets to be specified within the Trie that point
// to the end of the string, and the Trie can be traversed up to the root to
// decode the string value.
//
// This allows this string table to be used just like a normal string table
// where a 32 bit offset is specified, albeit with extra decoding needed to get
// the re-create the string values. During testing of some large C++ binaries,
// the trie string table was 45% of the size of the original string table. A
// DWARF optimizer could modify existing DWARF to switch over to using this new
// string table without having to re-write the stream of bytes in DWARF since
// a 32 bit string table offsets for both tables are the same size. A new DWARF
// form would need to be added to support this. Something like:
//
//  DW_FORM_strp_trie
//
// So a DWARF optimizer could easily re-write the .debug_abbrev section to
// replace any DW_FORM_strp entries with DW_FORM_strp_trie, replace the existing
// .debug_str section a new optimized .debug_str_trie and then fixup all
// previous DW_FORM_strp values with updated DW_FORM_strp_trie offsets.
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_TRIESTRTAB_H
#define LLDB_UTILITY_TRIESTRTAB_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include "lldb/Utility/DataExtractor.h"
#include "lldb/lldb-types.h"

#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

struct TrieNode;

namespace llvm {
namespace gsym {
class FileWriter;
}
} // namespace llvm

namespace lldb_private {

// A class that tracks the original string with optional offset. The original
// offset helps maps old string offsets to new string offsets in the Trie.
struct StringInfo {
  llvm::StringRef str;
  std::optional<uint32_t> orig_offset;

  bool operator<(const StringInfo &rhs) const {
    return str < rhs.str;
  }
};

class TrieBuilder {

public:
  ~TrieBuilder();
  /// Add a string reference to the string table.
  ///
  /// The strings added with this method are not copied and the lifetime of the
  /// strings must exceed the lifetime of this object.
  void AddString(StringInfo s) {
    m_string_refs.push_back(s);
    m_raw_strtab_size += s.str.size() + 1; // Include size with NULL
  }
  /// Add a string to the string table.
  ///
  /// The strings added with this method are copied and the copy is owned by
  /// this object.
  void AddString(std::string &&s);

  // Returns a bool true if success.
  bool Encode(llvm::gsym::FileWriter &file);
  const std::vector<lldb::offset_t> &GetStringOffsets() const {
    return m_str_offsets;
  }
  /// Load all strings from the specified string table file at the specified
  /// offset and size.
  ///
  /// The data in the string table must be one or more NULL terminated C-strings
  /// as found in the DWARF .debug_str section or ELF string table sections.
  ///
  /// \return Number of strings added to the trie string table.
  size_t AddStringsFromFile(const char *path, lldb::offset_t offset = 0,
                            lldb::offset_t size = UINT64_MAX);

  /// Load all strings from the specified string table data.
  ///
  /// The data must be one or more NULL terminated C-strings as found in the
  /// DWARF .debug_str section or ELF string table sections.
  ///
  /// \param[in] data The data to extract the strings from. This data is must
  ///   live longer than this class as references to the strings in the data
  ///   will be made and used throughout the lifetime of this object.
  ///
  /// \return Number of strings added to the trie string table.
  size_t AddStringsFromData(const lldb_private::DataExtractor &data);

  void Dump(llvm::ArrayRef<uint8_t> bytes) const;
  void DumpStats() const;

private:
  bool Build();
  TrieNode *MakeNode(TrieNode *parent, llvm::StringRef edge_str, bool terminal);
  void MakeEmptyTerminalNode(TrieNode *parent, size_t edge_idx);
  void BuildImpl(llvm::ArrayRef<StringInfo> vec, TrieNode *parent,
                 const size_t edge_start);
  std::vector<StringInfo> m_string_refs;
  std::map<uint32_t, uint32_t> m_orig_offset_to_new_offset;
  std::set<std::string> m_string_storage;
  std::vector<TrieNode *> m_nodes;
  std::vector<lldb::offset_t> m_str_offsets;
  uint64_t m_raw_strtab_size = 0;
  uint64_t m_encoded_size = 0;
};

class TrieStrtab {
  lldb_private::DataExtractor m_data;

public:
  TrieStrtab(const void *buf, size_t size, lldb::ByteOrder byte_order)
      : m_data(buf, size, byte_order, 4) {}

  llvm::Expected<std::string> GetString(lldb::offset_t str_offset);

private:
  llvm::Error GetStringImpl(lldb::offset_t offset,
                            std::optional<size_t> edge_idx,
                            std::vector<llvm::StringRef> &reversed_parts,
                            std::unordered_set<lldb::offset_t> &visited);
};

} // namespace lldb_private

#endif // #ifndef LLDB_UTILITY_TRIESTRTAB_H
