//===- TrieStrtab.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/TrieStrtab.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/Support/LEB128.h"
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <string>

using namespace llvm;
using namespace lldb_private;

struct Edge {
  Edge(StringRef s, TrieNode *node) : substring(s), child(node) {}
  StringRef substring;
  struct TrieNode *child;
};

struct TrieNode {
  static uint64_t total_parent_offset_size;
  static uint64_t total_parent_offsets_count;
  static uint64_t max_parent_node_offset;
  static uint64_t max_parent_node_offset_size;
  static uint64_t max_parent_edge_idx;
  static uint64_t max_edge_count;
  TrieNode(TrieNode *p, size_t i, bool t)
      : parent(p), parent_edge_idx(i), terminal(t) {}
  TrieNode *parent = nullptr;
  size_t parent_edge_idx = 0;
  std::vector<Edge> edges;
  bool terminal = 0;
  mutable std::optional<lldb::offset_t> encoded_offset;
  mutable std::optional<lldb::offset_t> encoded_end_offset;
  void Encode(llvm::gsym::FileWriter &file) const;
  void Dump(llvm::ArrayRef<uint8_t> bytes) const;
  int GetDepth() const {
    if (parent)
      return parent->GetDepth() + 1;
    return 0;
  }
};

uint64_t TrieNode::total_parent_offset_size = 0;
uint64_t TrieNode::total_parent_offsets_count = 0;
uint64_t TrieNode::max_parent_node_offset = 0;
uint64_t TrieNode::max_parent_node_offset_size = 0;
uint64_t TrieNode::max_parent_edge_idx = 0;
uint64_t TrieNode::max_edge_count = 0;

void TrieNode::Encode(llvm::gsym::FileWriter &file) const {
  encoded_offset = file.tell();
  // Encode the offset to subtract from this node's encoded_offset to get to
  // the parent Node. If offset is zero, then there is no parent.
  if (parent) {
    assert(parent->encoded_offset.has_value());
    const lldb::offset_t parent_node_offset =
        *encoded_offset - *parent->encoded_offset;
    file.writeULEB(parent_node_offset);
    if (max_parent_node_offset < parent_node_offset)
      max_parent_node_offset = parent_node_offset;
    const size_t parent_node_offset_size = file.tell() - *encoded_offset;
    if (max_parent_node_offset_size < parent_node_offset_size)
      max_parent_node_offset_size = parent_node_offset_size;
    total_parent_offset_size += parent_node_offset_size;
    ++total_parent_offsets_count;
    file.writeULEB(parent_edge_idx);
    if (parent_edge_idx > max_parent_edge_idx)
      max_parent_edge_idx = parent_edge_idx;
  } else {
    file.writeU8(0); // This is a ULEB but it fits into a byte.
  }

  const size_t num_edges = edges.size();
  if (num_edges > max_edge_count)
    max_edge_count = num_edges;
  // bool indicating if this is terminal node.
  if (num_edges < 127) {
    // Emit both the edge count plus 1 and the terminal bool as one value. The
    // edge count can be zero, so we add one to the edge count so we can tell
    //
    file.writeU8((num_edges + 1) << 1 | terminal);
  } else {
    // Edge count is >= 127, so we need to emit the terminal bool and edge count
    // separately
    file.writeU8(terminal);
    // Add number of children.
    file.writeULEB(num_edges);
  }
  // Append each child edge substring and any data for needed for the edge.
  for (const Edge &edge : edges) {
    file.writeNullTerminated(edge.substring);
    // // Encode the offset to subtract from the byte immediately after the
    // // edge.substring to get to the encoded offset of this node.
    // const lldb::offset_t offset = file.tell() - *encoded_offset;
    // file.writeULEB(offset);
  }
  encoded_end_offset = file.tell();
}

void TrieNode::Dump(llvm::ArrayRef<uint8_t> bytes) const {
  int depth = GetDepth();
  StringRef node_name;
  if (parent)
    node_name = parent->edges[parent_edge_idx].substring;
  if (encoded_offset)
    printf("0x%" PRIx64 ": ", *encoded_offset);
  printf("%*s\"%s\" terminal=%i\n", depth * 2, "", node_name.str().c_str(),
         terminal);
  if (encoded_offset && !bytes.empty()) {
    for (size_t i = *encoded_offset; i < *encoded_end_offset; ++i)
      printf("%2.2x ", bytes[i]);
    puts("");
  }

  for (const auto &edge : edges)
    edge.child->Dump(bytes);
}

TrieBuilder::~TrieBuilder() {
  for (TrieNode *node : m_nodes)
    delete node;
}

TrieNode *TrieBuilder::MakeNode(TrieNode *parent, StringRef edge_str,
                                bool terminal) {
  size_t edge_idx = 0;
  if (parent)
    edge_idx = parent->edges.size();
  auto *node = new TrieNode(parent, edge_idx, terminal);
  if (parent)
    parent->edges.emplace_back(edge_str, node);
  m_nodes.emplace_back(node);
  return node;
}

static std::optional<char> GetChar(StringRef str, size_t pos) {
  if (pos >= str.size())
    return std::nullopt;
  return str[pos];
}
static size_t GetAllMatchingChars(ArrayRef<llvm::StringRef> strs, size_t start,
                                  bool &terminal) {
  assert(!strs.empty());
  if (strs.size() == 1) {
    terminal = true;
    return strs.front().size();
  }
  size_t i = start;
  bool done = false;
  while (!done) {
    std::optional<char> ch1 = GetChar(strs.front(), i);
    if (ch1.has_value()) {
      const size_t num_strings = strs.size();
      size_t str_idx = 1;
      for (; str_idx < num_strings; ++str_idx) {
        std::optional<char> ch2 = GetChar(strs[str_idx], i);
        terminal = !ch2.has_value();
        if (ch1 != ch2)
          break;
      }
      if (str_idx == num_strings)
        ++i;
      else
        done = true;
    } else {
      terminal = true;
      done = true;
    }
  }
  return i;
}

void TrieBuilder::BuildImpl(ArrayRef<llvm::StringRef> strs, TrieNode *parent,
                            const size_t edge_start) {
  if (strs.empty())
    return;

  size_t end_str_idx = 1;
  size_t edge_end = edge_start;
  bool terminal = false;
  if (strs.size() == 1) {
    // Only one string, add the rest of the string and create a terminal node.
    terminal = true;
    StringRef edge_str(strs.front().substr(edge_start));
    MakeNode(parent, edge_str, terminal);
    return;
  }

  // Get the end index in the strs array whose first char matches.
  std::optional<char> ch1 = GetChar(strs.front(), edge_end);
  if (ch1.has_value()) {
    const size_t num_strings = strs.size();
    for (; end_str_idx < num_strings; ++end_str_idx) {
      std::optional<char> ch2 = GetChar(strs[end_str_idx], edge_end);
      terminal = !ch2.has_value();
      if (ch1 != ch2)
        break;
    }
    ++edge_end;
  } else {
    ++edge_end;
    terminal = true;
  }

  assert(edge_end > edge_start);
  ArrayRef<StringRef> common_prefix_strs = strs.slice(0, end_str_idx);
  ArrayRef<StringRef> non_common_prefix_strs = strs.slice(end_str_idx);
  assert(!common_prefix_strs.empty());
  // Match as many characters as possible within the common_prefix_strs
  if (!terminal)
    edge_end = GetAllMatchingChars(common_prefix_strs, edge_end, terminal);

  // Make a new child node for the current common_prefix_strs
  StringRef edge_str(strs.front().substr(edge_start, edge_end - edge_start));
  TrieNode *child = MakeNode(parent, edge_str, terminal);
  // assert(!terminal || common_prefix_strs.front().size() == edge_end);
  if (terminal)
    BuildImpl(common_prefix_strs.slice(1), child, edge_end);
  else
    BuildImpl(common_prefix_strs, child, edge_end);
  BuildImpl(non_common_prefix_strs, parent, edge_start);
}

bool TrieBuilder::Build() {
  if (m_string_refs.empty())
    return false;

  TrieNode *root = MakeNode(nullptr, /*edge_str=*/{}, /*terminal=*/false);
  llvm::sort(m_string_refs);
  BuildImpl(m_string_refs, root, /*edge_start=*/0);
  return true;
}

bool TrieBuilder::Encode(llvm::gsym::FileWriter &file) {
  if (Build()) {
    for (TrieNode *node : m_nodes) {
      node->Encode(file);
      // If the node is terminal, then add the offset of
      // the node to the string offsets as it represents
      // a unique string in the string table.
      if (node->terminal)
        m_str_offsets.push_back(*node->encoded_offset);
    }
    m_encoded_size = file.tell();
    return true;
  }
  return false;
}

void TrieBuilder::AddString(std::string &&s) {
  auto pair = m_string_storage.emplace(s);
  if (pair.second) {
    // String hasn't been added yet, add it.
    if (!pair.first->empty())
      AddStringRef(StringRef(*pair.first));
  } else {
    printf("duplicate string found in string table: \"%s\"\n", s.c_str());
  }
}

size_t TrieBuilder::AddStringsFromFile(const char *path, lldb::offset_t offset,
                                       lldb::offset_t size) {
  std::ifstream file(path, std::ios::binary); // Open file in binary mode
  if (!file.is_open())
    return 0;
  const size_t pre_existing_num_strings = m_string_refs.size();
  if (offset > 0)
    file.seekg(offset, std::ios::beg);
  std::optional<std::streampos> end_offset;
  if (size != UINT64_MAX)
    end_offset = offset + size;
  std::string s;
  while (std::getline(file, s, '\0')) {
    if (end_offset && file.tellg() >= *end_offset)
      break;
    AddString(std::move(s));
  }
  return m_string_refs.size() - pre_existing_num_strings;
}

size_t
TrieBuilder::AddStringsFromData(const lldb_private::DataExtractor &data) {
  const size_t pre_existing_num_strings = m_string_refs.size();
  lldb::offset_t offset = 0;
  while (data.ValidOffsetForDataOfSize(offset, 1)) {
    StringRef str(data.GetCStr(&offset));
    AddStringRef(str);
  }
  return m_string_refs.size() - pre_existing_num_strings;
}

void TrieBuilder::DumpStats() const {
  printf("Number of strings: %zu\n", m_string_refs.size());
  printf("Number of nodes: %zu\n", m_nodes.size());
  printf("Normal strtab size = %" PRIu64 "\n", m_raw_strtab_size);
  printf("Trie strtab size = %" PRIu64 " (%.2f%% smaller)\n", m_encoded_size,
         (1.0 - (float)m_encoded_size / (float)m_raw_strtab_size) * 100.0);
  float avg_parent_offset_size = (float)TrieNode::total_parent_offset_size /
                                 (float)TrieNode::total_parent_offsets_count;
  printf("max_parent_offset_size = %" PRIu64 " (average = %.2f)\n",
         TrieNode::max_parent_node_offset_size, avg_parent_offset_size);
  printf("max_parent_edge_idx = %" PRIu64 "\n", TrieNode::max_parent_edge_idx);
  printf("max_edge_count = %" PRIu64 "\n", TrieNode::max_edge_count);
}

void TrieBuilder::Dump(llvm::ArrayRef<uint8_t> bytes) const {
  if (m_nodes.empty())
    puts("empty TrieBuilder");
  else {
    m_nodes.front()->Dump(bytes);
  }
}

llvm::Expected<std::string> TrieStrtab::GetString(lldb::offset_t str_offset) {
  std::unordered_set<lldb::offset_t> visited;
  std::vector<StringRef> parts;
  llvm::Error error = GetStringImpl(str_offset, std::nullopt, parts, visited);
  if (error)
    return error;
  std::string str;
  for (auto iter = parts.rbegin(); iter != parts.rend(); ++iter)
    str += iter->str();
  return str;
}

llvm::Error
TrieStrtab::GetStringImpl(lldb::offset_t offset, std::optional<size_t> edge_idx,
                          std::vector<StringRef> &reversed_parts,
                          std::unordered_set<lldb::offset_t> &visited) {
  // Need at least 3 bytes:
  // - one for the parent offset ULEB128
  // - one for the parent edge index ULEB128
  // - one for "terminal" bool value + num edges
  if (!m_data.ValidOffsetForDataOfSize(offset, 3))
    return llvm::createStringError("invalid trie node offset 0x%" PRIx64,
                                   offset);

  const lldb::offset_t node_offset = offset;
  visited.insert(node_offset);
  const lldb::offset_t parent_reverse_offset = m_data.GetULEB128(&offset);
  const size_t parent_edge_idx =
      parent_reverse_offset > 0 ? m_data.GetULEB128(&offset) : 0;
  // If the edge count is <= 127, then the terminal bool and edge count are
  // encoded in 1 byte. If the edge count is > 127 then we will have one byte
  // for the terminal and ULEB128 for the edge count.
  const uint8_t terminal_and_edge_count = m_data.GetU8(&offset);
  if (edge_idx) {
    const uint64_t numEdges = terminal_and_edge_count <= 1
                                  ? m_data.GetULEB128(&offset)
                                  : (terminal_and_edge_count >> 1) - 1;
    for (uint64_t i = 0; i < numEdges; ++i) {
      StringRef str_part(m_data.GetCStr(&offset));
      if (i == edge_idx) {
        reversed_parts.push_back(str_part);
        break;
      }
    }
  }
  // If the parent_reverse_offset is zero, then it is the root node.
  if (parent_reverse_offset != 0) {
    const lldb::offset_t parent_offset = node_offset - parent_reverse_offset;
    if (visited.find(parent_offset) != visited.end())
      return llvm::createStringError("infinite loop in trie");
    llvm::Error error =
        GetStringImpl(parent_offset, parent_edge_idx, reversed_parts, visited);
    if (error)
      return error;
  }
  visited.erase(node_offset);
  return llvm::Error::success();
}
