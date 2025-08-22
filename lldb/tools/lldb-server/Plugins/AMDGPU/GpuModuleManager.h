//===-- GpuModuleManager.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file defines the GpuModuleManager class, which is used to track the
// state of loaded gpu modules (amd calls these "code objects").
//
// The state management is a bit complicated because the gpu debug library only
// returns the full set of loaded code objects, but lldb wants to be able
// to just get the list of modules that have changed. Additionally, we need
// to be able to track when a module has been unloaded. There is no separate
// event for that so we have to track it by detecting when a previously loaded
// module is no longer in the current list of active code objects returned by
// the debug library. The manager class hides those details and provides a
// convenient interface that works well to bridge the gap between lldb and the
// debug library.
//
// The class is designed so that we can process multiple events that signal
// changes to the code object list between reporting them to lldb. That is,
// we do not have to have a 1:1 mapping between calls to
// amd_dbgapi_process_code_object_list and the GetGPUDynamicLoaderLibraryInfos
// callback from lldb.
//
// One particular design decision was to use a SetVector to store the set of
// currently loaded code objects. This was done to ensure we have a
// deterministic order of modules when reporting them to lldb regardless of the
// additions or removals from the set.
//
//===----------------------------------------------------------------------===//
#include "lldb/lldb-types.h"
#include "llvm/ADT/SetVector.h"
#include <string>
#include <unordered_set>
#include <vector>

namespace lldb_private {
namespace lldb_server {

class GpuModuleManager {
public:
  // A code object as reported by the debug library.
  // It has a uri (see AMD_DBGAPI_CODE_OBJECT_INFO_URI_NAME) and a load address.
  // The debug library only reports loaded code objects, but we use the state
  // to track when an object has been unloaded.
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

    // Hash function so we can store it in a set.
    struct Hasher {
      std::size_t operator()(const CodeObject &obj) const {
        std::size_t h1 = std::hash<std::string>{}(obj.uri);
        std::size_t h2 = std::hash<lldb::addr_t>{}(obj.load_address);
        return h1 ^ (h2 << 1);
      }
    };
  };
  typedef std::vector<CodeObject> CodeObjectList;

  // Call this at the start of processing all the code objects
  // returned from amd_dbgapi_process_code_object_list.
  void BeginCodeObjectListUpdate() {
    // Reset the state for this update.
    m_update_alive.clear();
    m_update_new.clear();
  }

  // Call this for each code object in the list returned by the debug library.
  void CodeObjectIsLoaded(const std::string &uri, lldb::addr_t addr) {
    // Mark this code object as alive.
    CodeObject obj{uri, addr};
    m_update_alive.insert(obj);

    // Add it as a new code object in this update if we have not seen it before.
    if (!m_code_objects.contains(obj)) {
      m_code_objects.insert(obj);
      m_update_new.emplace_back(obj);
    }
  }

  // Call this after processing all of the code objects.
  void EndCodeObjectListUpdate() {
    // Remove any code object that is not alive.
    // Any removed object is added to the changes list as unloaded.
    m_code_objects.remove_if([this](const CodeObject &obj) {
      if (!m_update_alive.count(obj)) {
        CodeObject unloaded_obj = obj;
        unloaded_obj.state = CodeObject::State::Unloaded;
        m_changes.emplace_back(unloaded_obj);
        return true;
      }
      return false;
    });

    // Add any new code objects to the change list.
    for (const CodeObject &obj : m_update_new)
      m_changes.emplace_back(obj);

    m_update_new.clear();
    m_update_alive.clear();
  }

  // Return the full set of loaded code objects.
  llvm::iterator_range<CodeObjectList::const_iterator>
  GetLoadedCodeObjects() const {
    return m_code_objects;
  }

  // Return the set of code objects since the last call to
  // `ClearChangedObjectList`.
  llvm::iterator_range<CodeObjectList::const_iterator>
  GetChangedCodeObjects() const {
    return m_changes;
  }

  // Reset the tracked changes.
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
