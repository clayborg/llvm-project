//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DisassemblerSASS.h"

#include "DisassemblerSASSCache.h"
#include "InstructionSASS.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <vector>

using namespace lldb;
using namespace lldb_private;
namespace json = llvm::json;

LLDB_PLUGIN_DEFINE(DisassemblerSASS)

static bool IsNvptxArch(const ArchSpec &arch) {
  return arch.GetTriple().getArch() == llvm::Triple::nvptx ||
         arch.GetTriple().getArch() == llvm::Triple::nvptx64;
}

DisassemblerSASS::DisassemblerSASS(const ArchSpec &arch, const char *flavor,
                                   const char *cpu, const char *features)
    : Disassembler(arch, flavor), m_valid(false) {

  Log *log = GetLog(LLDBLog::Disassembler);

  // Check if this is an NVPTX architecture (which produces SASS when compiled)
  if (!IsNvptxArch(arch)) {
    LLDB_LOG(log, "DisassemblerSASS: Unsupported architecture: {0}",
             arch.GetTriple().getArchName());
    return;
  }

  // Try to find nvdisasm
  if (llvm::Expected<FileSpec> nvdisasm_or =
          DisassemblerSASSCache::getInstance().GetNvdisasmPath()) {
    m_nvdisasm_path = *nvdisasm_or;
    m_valid = true;
    LLDB_LOG(log,
             "DisassemblerSASS: Successfully initialized with nvdisasm at: {0}",
             m_nvdisasm_path.GetPath());
  } else {
    m_valid = false;
    Debugger::ReportError(llvm::toString(nvdisasm_or.takeError()));
  }
}

DisassemblerSASS::~DisassemblerSASS() = default;

lldb::DisassemblerSP DisassemblerSASS::CreateInstance(const ArchSpec &arch,
                                                      const char *flavor,
                                                      const char *cpu,
                                                      const char *features) {
  if (IsNvptxArch(arch)) {
    auto disasm_sp =
        std::make_shared<DisassemblerSASS>(arch, flavor, cpu, features);
    if (disasm_sp && disasm_sp->IsValid())
      return disasm_sp;
  }
  return lldb::DisassemblerSP();
}

size_t DisassemblerSASS::DecodeInstructions(const Address &base_addr,
                                            const DataExtractor &data,
                                            lldb::offset_t data_offset,
                                            size_t num_instructions,
                                            bool append, bool data_from_file) {

  Log *log = GetLog(LLDBLog::Disassembler);
  if (!IsValid()) {
    LLDB_LOG(log, "DisassemblerSASS::DecodeInstructions: Cannot disassemble - "
                  "nvdisasm not available");

    Debugger::ReportError(
        "nvdisasm not found. Please install CUDA toolkit or set "
        "CUDA_HOME environment variable.");
    return 0;
  }

  if (!append)
    m_instruction_list.Clear();

  LLDB_LOG(log,
           "DisassemblerSASS::DecodeInstructions: base_addr={0:x}, "
           "data_offset={1}, num_instructions={2}",
           base_addr.GetFileAddress(), data_offset, num_instructions);

  if (llvm::Expected<size_t> result_or =
          DisassembleWithNvdisasm(data, base_addr, num_instructions)) {
    return *result_or;
  } else {
    std::string error_msg = llvm::toString(result_or.takeError());
    LLDB_LOG(log, "DisassembleWithNvdisasm failed: {0}", error_msg);

    // Report error to debugger
    Debugger::ReportError(error_msg);
    return 0;
  }
}

void DisassemblerSASS::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "Disassembler that uses nvdisasm to "
                                "disassemble SASS (Shader Assembly) code.",
                                CreateInstance);
}

void DisassemblerSASS::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

bool DisassemblerSASS::FlavorValidForArchSpec(
    const lldb_private::ArchSpec &arch, const char *flavor) {
  // Accept any flavor for NVPTX or default to "default"
  return IsNvptxArch(arch) &&
         (flavor == nullptr || llvm::StringRef(flavor) == "default");
}

llvm::Expected<std::string>
DisassemblerSASS::ExtractSmArchFromModule(const Address &base_addr) {
  lldb::ModuleSP module_sp = base_addr.GetModule();
  if (!module_sp)
    return llvm::createStringError("No module found for address");

  // Use singleton cache to get the per-module SM extractor
  DisassemblerSASSCache &cache = DisassemblerSASSCache::getInstance();

  // Generate cache key for in-memory modules
  // For memory-only modules, prefer UUID, fallback to module pointer
  std::string cache_key;
  if (module_sp->GetUUID().IsValid())
    cache_key = module_sp->GetUUID().GetAsString();
  else {
    // Fallback to module pointer for modules without UUID
    cache_key = llvm::formatv("module_ptr_{0:x}",
                              reinterpret_cast<uintptr_t>(module_sp.get()))
                    .str();
  }
  Log *log = GetLog(LLDBLog::Disassembler);
  LLDB_LOG(log, "ExtractSmArchFromModule: Cache key: {0}", cache_key);

  std::shared_ptr<ModuleSM> module_sm = cache.GetModuleSM(cache_key);
  return module_sm->FindSM(base_addr);
}

/// Invoke nvdisasm and return the JSON output.
static llvm::Expected<std::string>
InvokeNVDisasm(const DataExtractor &data, const std::string &sm_arch,
               const FileSpec &nvdisasm_path) {
  Log *log = GetLog(LLDBLog::Disassembler);

  // Create a temporary file for the binary data
  llvm::SmallString<128> temp_file_path;
  int temp_fd;
  std::error_code ec = llvm::sys::fs::createTemporaryFile(
      "lldb_sass", "cubin", temp_fd, temp_file_path);
  if (ec)
    return llvm::createStringError(
        ec, "Failed to create temporary sass file for nvdisasm");

  // RAII cleanup for temporary cubin file
  llvm::FileRemover temp_file_remover(temp_file_path);

  // Write the binary data to the temporary file.
  {
    llvm::raw_fd_ostream file_stream(temp_fd, /*shouldClose=*/true);
    file_stream.write(reinterpret_cast<const char *>(data.GetDataStart()),
                      data.GetByteSize());

    if (file_stream.has_error())
      return llvm::createStringError("Failed to write data to temporary file");
  }

  std::string nvdisasm_path_str = nvdisasm_path.GetPath();

  // Prepare arguments for nvdisasm (including program name as argv[0])
  llvm::SmallVector<llvm::StringRef, 8> args;
  args.push_back(nvdisasm_path_str);
  args.push_back("-b");
  args.push_back(sm_arch.c_str()); // Use extracted SM architecture
  args.push_back("--emit-json");
  args.push_back(temp_file_path.c_str());

  LLDB_LOG(log, "Running nvdisasm: {0} {1} {2} {3} {4}", args[0], args[1],
           args[2], args[3], args[4]);

  // Create temporary files for stdout and stderr redirection
  llvm::SmallString<128> stdout_path, stderr_path;
  std::error_code stdout_ec =
      llvm::sys::fs::createTemporaryFile("nvdisasm_stdout", "", stdout_path);
  if (stdout_ec) {
    return llvm::createStringError(
        stdout_ec, "Failed to create temporary output file for nvdisasm");
  }

  // RAII cleanup for stdout temporary file
  llvm::FileRemover stdout_file_remover(stdout_path);

  std::error_code stderr_ec =
      llvm::sys::fs::createTemporaryFile("nvdisasm_stderr", "", stderr_path);
  if (stderr_ec) {
    return llvm::createStringError(
        stderr_ec, "Failed to create temporary error file for nvdisasm");
  }

  // RAII cleanup for stderr temporary file
  llvm::FileRemover stderr_file_remover(stderr_path);

  std::array<std::optional<llvm::StringRef>, 3> redirects = {
      std::nullopt,                 // stdin
      llvm::StringRef(stdout_path), // stdout
      llvm::StringRef(stderr_path)  // stderr
  };

  int exit_code = llvm::sys::ExecuteAndWait(nvdisasm_path_str, args,
                                            /*env=*/std::nullopt, redirects);

  // Read stdout and stderr
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> stdout_buffer =
      llvm::MemoryBuffer::getFile(stdout_path);
  if (!stdout_buffer) {
    return llvm::createStringError(stdout_buffer.getError(),
                                   "Failed to read nvdisasm stdout");
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> stderr_buffer =
      llvm::MemoryBuffer::getFile(stderr_path);

  if (exit_code != 0) {
    std::string stderr_str;
    if (stderr_buffer)
      stderr_str = stderr_buffer.get()->getBuffer().str();
    return llvm::createStringError("nvdisasm failed with exit code %d: %s",
                                   exit_code, stderr_str.c_str());
  }

  std::string json_output = stdout_buffer.get()->getBuffer().str();

  LLDB_LOG(log, "nvdisasm output: {0}", json_output);
  return json_output;
}

llvm::Expected<size_t>
DisassemblerSASS::DisassembleWithNvdisasm(const DataExtractor &data,
                                          const Address &base_addr,
                                          size_t max_instructions) {
  Log *log = GetLog(LLDBLog::Disassembler);

  if (!IsValid()) {
    LLDB_LOG(log, "DisassemblerSASS is not valid - nvdisasm not available");
    return llvm::createStringError("nvdisasm not available");
  }

  std::string sm_arch;
  if (llvm::Expected<std::string> sm_arch_or =
          ExtractSmArchFromModule(base_addr)) {
    sm_arch = *sm_arch_or;
  } else {
    LLDB_LOG(log, "Failed to extract SM architecture: {0}",
             llvm::toString(sm_arch_or.takeError()));
    // TODO DTCLLDB-63: Add a fallback here that the user can configure
    return llvm::createStringError(
        "Failed to extract SM architecture from CUDA module");
  }

  std::string json_output;
  if (llvm::Expected<std::string> json_output_or =
          InvokeNVDisasm(data, sm_arch, m_nvdisasm_path)) {
    json_output = std::move(*json_output_or);
  } else {
    return json_output_or.takeError();
  }

  if (llvm::Expected<size_t> parse_result_or =
          ParseNvdisasmJsonOutput(json_output, base_addr, max_instructions)) {
    LLDB_LOG(log, "Parsed {0} instructions", *parse_result_or);
    return *parse_result_or;
  } else {
    return parse_result_or.takeError();
  }
}

llvm::Expected<size_t>
DisassemblerSASS::ParseNvdisasmJsonOutput(const std::string &json_output,
                                          const Address &base_addr,
                                          size_t max_instructions) {
  Log *log = GetLog(LLDBLog::Disassembler);
  const size_t instruction_size = InstructionSASS::GetInstructionByteSize();

  llvm::Expected<json::Value> json_value = json::parse(json_output);
  if (!json_value) {
    LLDB_LOG(log, "Failed to parse JSON output from nvdisasm: {0}",
             llvm::toString(json_value.takeError()));
    return llvm::createStringError("Failed to parse JSON output from nvdisasm");
  }

  // The output is an array with two elements: [metadata, functions_array]
  const json::Array *root_array = json_value->getAsArray();
  if (!root_array || root_array->size() < 2) {
    LLDB_LOG(log,
             "JSON output is not an array or has insufficient elements. "
             "Expected: [metadata, functions_array], got size: {0}",
             root_array ? root_array->size() : 0);
    return llvm::createStringError(
        "JSON output is not an array or has insufficient elements");
  }

  // Validate metadata structure (first element)
  const json::Object *metadata = (*root_array)[0].getAsObject();
  if (!metadata)
    return llvm::createStringError("First element (metadata) is not an object");

  // Log some metadata information if available
  if (log) {
    if (std::optional<llvm::StringRef> producer =
            metadata->getString("Producer"))
      LLDB_LOG(log, "nvdisasm producer: {0}", producer->str());
    if (const json::Object *sm_obj = metadata->getObject("SM"))
      if (const json::Object *version_obj = sm_obj->getObject("version"))
        if (std::optional<int64_t> major = version_obj->getInteger("major"))
          if (std::optional<int64_t> minor = version_obj->getInteger("minor"))
            LLDB_LOG(log, "Target SM version: {0}.{1}", *major, *minor);
  }

  // The second element contains the functions array
  const json::Array *functions_array = (*root_array)[1].getAsArray();
  if (!functions_array)
    return llvm::createStringError("Second element is not an array");

  LLDB_LOG(log, "Found {0} functions in nvdisasm output",
           functions_array->size());

  size_t instructions_parsed = 0;

  for (const json::Value &func_value : *functions_array) {
    const json::Object *func_obj = func_value.getAsObject();
    if (!func_obj)
      continue;

    std::optional<llvm::StringRef> function_name =
        func_obj->getString("function-name");
    std::optional<int64_t> start_addr = func_obj->getInteger("start");
    std::optional<int64_t> length = func_obj->getInteger("length");

    if (!start_addr) {
      LLDB_LOG(log, "Function missing required 'start' field");
      continue;
    }

    LLDB_LOG(log, "Processing function: name='{0}', start={1:x}, length={2}",
             function_name ? function_name->str() : "<unknown>", *start_addr,
             length ? *length : 0);

    const json::Array *sass_instructions =
        func_obj->getArray("sass-instructions");
    if (!sass_instructions) {
      LLDB_LOG(log, "Function has no 'sass-instructions' array");
      continue;
    }

    size_t instruction_offset = 0;
    for (const json::Value &inst_value : *sass_instructions) {
      if (instructions_parsed >= max_instructions)
        break;

      const json::Object *inst_obj = inst_value.getAsObject();
      if (!inst_obj) {
        LLDB_LOG(log, "Gotten an Instruction entry from nvdisasm that is not "
                      "an object. Stopping disassembly.");
        break;
      }

      std::optional<llvm::StringRef> opcode_val = inst_obj->getString("opcode");
      std::optional<llvm::StringRef> operands_val =
          inst_obj->getString("operands");
      std::optional<llvm::StringRef> predicate_val =
          inst_obj->getString("predicate");
      std::optional<llvm::StringRef> extra_val = inst_obj->getString("extra");

      if (!opcode_val) {
        LLDB_LOG(log,
                 "Missing opcode entry from nvdisasm. Stopping disassembly.");
        break;
      }

      // Create address for this instruction
      // nvdisasm 'start' field represents offset within the disassembled
      // region.
      Address inst_addr(base_addr);
      addr_t calculated_offset =
          *start_addr + instruction_offset * instruction_size;

      LLDB_LOG(log,
               "Address calculation: base_addr={0:x}, start_addr={1:x}, "
               "instruction_offset={2}, calculated_offset={3:x}",
               base_addr.GetFileAddress(), *start_addr, instruction_offset,
               calculated_offset);

      inst_addr.Slide(calculated_offset);

      // Get optional fields
      std::string operands = operands_val ? operands_val->str() : "";
      std::string predicate = predicate_val ? predicate_val->str() : "";
      std::string extra = extra_val ? extra_val->str() : "";

      // Extract other-attributes
      std::map<std::string, std::string> other_attributes;
      if (const json::Object *attrs = inst_obj->getObject("other-attributes")) {
        for (const auto &attr : *attrs) {
          if (std::optional<llvm::StringRef> s = attr.second.getAsString())
            other_attributes[attr.first.str()] = s->str();
          else if (std::optional<bool> b = attr.second.getAsBoolean())
            other_attributes[attr.first.str()] = *b ? "True" : "False";
          else {
            // Fallback for integers / nulls
            other_attributes[attr.first.str()] =
                llvm::formatv("{0}", attr.second).str();
          }
        }
      }

      std::vector<std::string> other_flags;
      if (const json::Array *flags = inst_obj->getArray("other-flags"))
        for (const json::Value &flag_val : *flags)
          if (std::optional<llvm::StringRef> flag_str = flag_val.getAsString())
            other_flags.push_back(flag_str->str());

      auto inst_sp = std::make_shared<InstructionSASS>(
          inst_addr, opcode_val->str(), operands, predicate, extra,
          other_attributes, other_flags, AddressClass::eCode);

      // Set dummy opcode bytes (we don't have the actual bytes from nvdisasm)
      // SASS instructions are 8 bytes each
      uint64_t dummy_opcode = 0;
      inst_sp->SetOpcode(&dummy_opcode, sizeof(dummy_opcode));

      lldb::InstructionSP instruction_sp =
          std::static_pointer_cast<lldb_private::Instruction>(inst_sp);
      m_instruction_list.Append(instruction_sp);
      instructions_parsed++;
      instruction_offset++;

      LLDB_LOG(log,
               "Created instruction: opcode='{0}', operands='{1}', "
               "predicate='{2}', extra='{3}', attributes={4}, flags={5} at "
               "addr={6:x}",
               opcode_val->str(), operands, predicate, extra,
               other_attributes.size(), other_flags.size(),
               inst_addr.GetFileAddress());
    }
  }

  if (instructions_parsed > 0) {
    InstructionSP last_inst = m_instruction_list.GetInstructionAtIndex(
        m_instruction_list.GetSize() - 1);

    return last_inst->GetAddress().GetFileAddress() -
           base_addr.GetFileAddress() + instruction_size;
  }

  return 0;
}
