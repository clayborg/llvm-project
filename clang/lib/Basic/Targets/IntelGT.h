//===--- IntelGT.h - Declare Intel GT GPU target feature support -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Intel GT (intelgt) TargetInfo. It is intentionally
// minimal.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_INTELGT_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_INTELGT_H

#include "Targets.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY IntelGTTargetInfo final : public TargetInfo {
public:
  IntelGTTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    assert(Triple.getArch() == llvm::Triple::intelgt &&
           "Invalid architecture for Intel GT.");
    // 64-bit little-endian GPU.
    PointerWidth = PointerAlign = 64;
    LongWidth = LongAlign = 64;
    SizeType = TargetInfo::UnsignedLong;
    PtrDiffType = IntPtrType = TargetInfo::SignedLong;
    // No TLS on GPU targets; no inline asm variants either.
    TLSSupported = false;
    VLASupported = false;
    NoAsmVariants = true;
    // Data layout mirrors the Intel SPIR-V variant (64-bit pointers, common
    // GPU vector alignments). This matters for Clang's ASTContext type-size
    // queries; LLDB does not JIT Intel Xe code from this TargetInfo.
    resetDataLayout("e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-"
                    "v256:256-v512:512-v1024:1024-n8:16:32:64-G1-P9-A0");
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  llvm::SmallVector<Builtin::InfosShard> getTargetBuiltins() const override {
    return {};
  }

  std::string_view getClobbers() const override { return ""; }

  ArrayRef<const char *> getGCCRegNames() const override { return {}; }

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return {};
  }

  bool validateAsmConstraint(const char *&,
                             TargetInfo::ConstraintInfo &) const override {
    return false;
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  bool hasFeature(StringRef Feature) const override {
    return Feature == "intelgt";
  }

  bool hasBitIntType() const override { return true; }
  bool hasInt128Type() const override { return false; }
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_INTELGT_H
