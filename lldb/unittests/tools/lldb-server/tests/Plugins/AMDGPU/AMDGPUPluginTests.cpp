//===-- AMDGPUPluginTests.cpp
//------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GpuModuleManager.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_server;
using namespace llvm;

class GpuModuleManagerTest : public ::testing::Test {
protected:
  using CodeObject = GpuModuleManager::CodeObject;
  using CodeObjectList = GpuModuleManager::CodeObjectList;
  GpuModuleManager m_gmm;
  CodeObjectList m_objs;
  CodeObjectList m_changed;

  // Makeup some fake code object uris.
  const std::string uri_1 = "file://a.out#offset=100&size=10";
  const std::string uri_2 = "file://a.out#offset=200&size=20";
  const std::string uri_3 = "file://a.out#offset=300&size=30";
  const std::string uri_4 = "file://a.out#offset=400&size=40";

  void SetUp() override {
    // Add a few objects we can use for most tests.
    m_objs = {
        {uri_1, 1000},
        {uri_2, 2000},
    };

    for (const CodeObject &obj : m_objs) {
      m_changed.push_back(WithState(obj, CodeObject::State::Loaded));
    }
  }

  void UpdateCodeObjectList(const GpuModuleManager::CodeObjectList &objs) {
    m_gmm.BeginCodeObjectListUpdate();
    for (const auto &obj : objs) {
      m_gmm.CodeObjectIsLoaded(obj.uri, obj.load_address);
    }
    m_gmm.EndCodeObjectListUpdate();
  }

  CodeObject RemoveObjectAtIndex(size_t index) {
    assert(index < m_objs.size());
    CodeObject obj = m_objs[index];
    m_objs.erase((m_objs.begin() + index));
    return obj;
  }

  CodeObject AppendCodeObject(const std::string &uri, lldb::addr_t addr) {
    CodeObject obj(uri, addr);
    m_objs.push_back(obj);
    return obj;
  }

  CodeObject WithState(const CodeObject &obj, CodeObject::State state) {
    CodeObject new_obj = obj;
    new_obj.state = state;
    return new_obj;
  }

  // Helper for easier testing to turn the range into a vector.
  CodeObjectList GetLoadedCodeObjects() {
    auto range = m_gmm.GetLoadedCodeObjects();
    return CodeObjectList(range.begin(), range.end());
  }

  // Helper for easier testing to turn the range into a vector.
  CodeObjectList GetChangedCodeObjects() {
    auto range = m_gmm.GetChangedCodeObjects();
    return CodeObjectList(range.begin(), range.end());
  }

  // Get the changed objects and clear the list.
  CodeObjectList ConsumeChangedCodeObjects() {
    CodeObjectList objs = GetChangedCodeObjects();
    m_gmm.ClearChangedObjectList();

    return objs;
  }
};

// Helper to pretty-print code objects in gtest messages.
namespace lldb_private::lldb_server {
void PrintTo(const GpuModuleManager::CodeObject &obj, std::ostream *os) {
  *os << "CodeObject(" << obj.uri << "," << obj.load_address << ")";
}
} // namespace lldb_private::lldb_server

TEST_F(GpuModuleManagerTest, TestEmptyUpdate) {
  UpdateCodeObjectList(CodeObjectList());

  // Validate there are no loaded objects.
  ASSERT_EQ(CodeObjectList(), GetLoadedCodeObjects());
  ASSERT_EQ(CodeObjectList(), GetChangedCodeObjects());
}

TEST_F(GpuModuleManagerTest, TestBasicUpdate) {
  UpdateCodeObjectList(m_objs);

  // Validate the loaded objects matches the expected list.
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());
  ASSERT_EQ(m_changed, ConsumeChangedCodeObjects());

  // The objects should all have the loaded state.
  for (const CodeObject &obj : GetLoadedCodeObjects()) {
    ASSERT_EQ(obj.state, CodeObject::State::Loaded);
    ASSERT_TRUE(obj.IsLoaded());
  }
}

TEST_F(GpuModuleManagerTest, TestClearChangedCodeObject) {
  UpdateCodeObjectList(m_objs);

  // The changed objects list should only reset when cleared.
  ASSERT_EQ(m_changed, GetChangedCodeObjects());
  ASSERT_TRUE(m_gmm.HasChangedCodeObjects());
  ASSERT_EQ(m_changed, GetChangedCodeObjects());
  m_gmm.ClearChangedObjectList();
  ASSERT_FALSE(m_gmm.HasChangedCodeObjects());
  ASSERT_EQ(CodeObjectList(), GetChangedCodeObjects());
}

TEST_F(GpuModuleManagerTest, TestNoopChangeList) {
  UpdateCodeObjectList(m_objs);

  // Validate the loaded objects matches the expected list.
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());
  ASSERT_EQ(m_changed, ConsumeChangedCodeObjects());

  UpdateCodeObjectList(m_objs);

  // Loading the same code objects the second time should produce
  // the same set of loaded code objects but an empty changed list
  // since there were no changes.
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());
  ASSERT_EQ(CodeObjectList(), GetChangedCodeObjects());
}

TEST_F(GpuModuleManagerTest, TestChangeListLoad) {
  UpdateCodeObjectList(m_objs);

  // Validate the loaded objects matches the expected list.
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());
  ASSERT_EQ(m_changed, ConsumeChangedCodeObjects());

  // Now add a new code object to the list.
  CodeObject obj{uri_3, 3000};
  CodeObject changed = WithState(obj, CodeObject::State::Loaded);
  m_objs.push_back(obj);
  UpdateCodeObjectList(m_objs);

  // Should have the full set of code objects and one load in changed.
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());
  ASSERT_EQ(CodeObjectList({changed}), GetChangedCodeObjects());
}

TEST_F(GpuModuleManagerTest, TestChangeListUnload) {
  UpdateCodeObjectList(m_objs);

  // Validate the loaded objects matches the expected list.
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());
  ASSERT_EQ(m_changed, ConsumeChangedCodeObjects());

  // Now remove a code object from the list.
  CodeObject obj = RemoveObjectAtIndex(0);
  CodeObject changed = WithState(obj, CodeObject::State::Unloaded);
  UpdateCodeObjectList(m_objs);

  // Should have the reduced set of code objects and one unload changed.
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());
  ASSERT_EQ(CodeObjectList({changed}), GetChangedCodeObjects());
}

TEST_F(GpuModuleManagerTest, TestChangeListLoadAndUnload) {
  UpdateCodeObjectList(m_objs);

  // Validate the loaded objects matches the expected list.
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());
  ASSERT_EQ(m_changed, ConsumeChangedCodeObjects());

  // Now remove and add a code object from the list.
  CodeObject unload_change =
      WithState(RemoveObjectAtIndex(0), CodeObject::State::Unloaded);
  CodeObject load_change =
      WithState(AppendCodeObject(uri_4, 4000), CodeObject::State::Loaded);

  UpdateCodeObjectList(m_objs);

  // Should have the new set of code objects with one unload
  // changed coming befor the load changed.
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());
  ASSERT_EQ(CodeObjectList({unload_change, load_change}),
            GetChangedCodeObjects());
}

TEST_F(GpuModuleManagerTest, TestChangedListAcrossUpdates) {
  UpdateCodeObjectList(m_objs);

  // Validate the loaded objects matches the expected list.
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());

  // Add a new code object and update the list.
  CodeObject obj = AppendCodeObject(uri_3, 3000);
  m_changed.push_back(WithState(obj, CodeObject::State::Loaded));

  UpdateCodeObjectList(m_objs);

  // Validate the loaded objects matches the expected list.
  // The changed should contain the full set of changes not just
  // the latest change.
  ASSERT_GT(m_changed.size(), 1);
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());
  ASSERT_EQ(m_changed, GetChangedCodeObjects());
}

TEST_F(GpuModuleManagerTest, TestChangedListLoadUnloadSameObject) {
  // Clear all internal state for this test.
  m_objs.clear();
  m_changed.clear();

  // Add one new code object.
  CodeObject obj = AppendCodeObject(uri_1, 1000);
  UpdateCodeObjectList(m_objs);
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());

  // Clear out all the loaded code objects.
  m_objs.clear();
  UpdateCodeObjectList(m_objs);
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());

  // Re-add the code object.
  m_objs.push_back(obj);
  UpdateCodeObjectList(m_objs);
  ASSERT_EQ(m_objs, GetLoadedCodeObjects());

  // The change list should show the load-unload-load sequence.
  CodeObjectList changed = {
      WithState(obj, CodeObject::State::Loaded),
      WithState(obj, CodeObject::State::Unloaded),
      WithState(obj, CodeObject::State::Loaded),
  };
  ASSERT_EQ(changed, GetChangedCodeObjects());
}
