add_subdirectory(Utils)

if(LLDB_ENABLE_AMDGPU_PLUGIN)
  add_subdirectory(AMDGPU)
endif()

if(LLDB_ENABLE_NVGPU_PLUGIN)
  add_subdirectory(NVGPU)
endif()

if(LLDB_ENABLE_MOCK_GPU_PLUGIN)
  add_subdirectory(MockGPU)
endif()
