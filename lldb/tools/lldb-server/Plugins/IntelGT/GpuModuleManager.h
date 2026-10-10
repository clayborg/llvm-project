//===-- GpuModuleManager.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_GPUMODULEMANAGER_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_GPUMODULEMANAGER_H

#include "lldb/lldb-types.h"
#include "llvm/ADT/SetVector.h"
#include <string>
#include <unordered_set>
#include <vector>

namespace lldb_private {
namespace lldb_server {

/// Tracks loaded GPU code objects and reports load/unload changes to lldb.
/// Unloads are detected as absence between successive snapshots.
class GpuModuleManager {
public:
  /// A code object with URI, load address, and current load state.
  struct CodeObject {
    enum State { Unloaded, Loaded };
    std::string uri;
    lldb::addr_t load_address = 0;
    State state = Loaded;

    CodeObject() = default;
    CodeObject(const std::string &u, lldb::addr_t addr)
        : uri(u), load_address(addr) {}

    bool IsLoaded() const { return state == State::Loaded; }

    bool operator==(const CodeObject &other) const {
      return uri == other.uri && load_address == other.load_address;
    }

    /// Hash function for storing in a set.
    struct Hasher {
      std::size_t operator()(const CodeObject &obj) const {
        std::size_t h1 = std::hash<std::string>{}(obj.uri);
        std::size_t h2 = std::hash<lldb::addr_t>{}(obj.load_address);
        return h1 ^ (h2 << 1);
      }
    };
  };
  typedef std::vector<CodeObject> CodeObjectList;

  /// Begin processing a full code-object snapshot.
  void BeginCodeObjectListUpdate() {
    m_update_alive.clear();
    m_update_new.clear();
  }

  /// Record a currently loaded code object.
  void CodeObjectIsLoaded(const std::string &uri, lldb::addr_t addr) {
    CodeObject obj{uri, addr};
    m_update_alive.insert(obj);

    // Add as new if we have not seen it before.
    if (!m_code_objects.contains(obj)) {
      m_code_objects.insert(obj);
      m_update_new.emplace_back(obj);
    }
  }

  /// Finish processing the snapshot; diffs generate load/unload changes.
  void EndCodeObjectListUpdate() {
    // Drop objects no longer alive and record them as unloaded.
    m_code_objects.remove_if([this](const CodeObject &obj) {
      if (!m_update_alive.count(obj)) {
        CodeObject unloaded_obj = obj;
        unloaded_obj.state = CodeObject::State::Unloaded;
        m_changes.emplace_back(unloaded_obj);
        return true;
      }
      return false;
    });

    for (const CodeObject &obj : m_update_new)
      m_changes.emplace_back(obj);

    m_update_new.clear();
    m_update_alive.clear();
  }

  /// Return the full set of loaded code objects.
  llvm::iterator_range<CodeObjectList::const_iterator>
  GetLoadedCodeObjects() const {
    return m_code_objects;
  }

  /// Return code objects changed since the last ClearChangedObjectList().
  llvm::iterator_range<CodeObjectList::const_iterator>
  GetChangedCodeObjects() const {
    return m_changes;
  }

  /// Reset the tracked changes.
  void ClearChangedObjectList() { m_changes.clear(); }

  bool HasChangedCodeObjects() const { return !m_changes.empty(); }

private:
  typedef std::unordered_set<CodeObject, CodeObject::Hasher> CodeObjectSet;
  typedef llvm::SetVector<CodeObject, CodeObjectList, CodeObjectSet>
      CodeObjects;
  CodeObjects m_code_objects;
  CodeObjectList m_changes;
  CodeObjectList m_update_new;
  CodeObjectSet m_update_alive;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_INTELGT_GPUMODULEMANAGER_H
