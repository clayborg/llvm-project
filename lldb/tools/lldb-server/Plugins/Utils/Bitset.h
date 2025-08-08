//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_UTILS_BITSET_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_UTILS_BITSET_H

#include <llvm/ADT/StringRef.h>
#include <vector>

namespace lldb_private::lldb_server {
/// A simple bitset implementation for managing byte arrays whose size is not
/// known at compile time.
///
/// This class does not check if the indices of getters and setters are valid to
/// guarantee maximum performance.
class DynamicBitset {
public:
  /// Construct a bitset from a byte array.
  ///
  /// \param[in] bytes
  ///     A StringRef containing the byte data to initialize the bitset.
  /// \param[in] size
  ///     The number of bits in the bitset.
  DynamicBitset(llvm::StringRef bytes, size_t size) : m_size(size) {
    m_bytes.resize((m_size + 7) / 8);
    size_t bytes_to_copy = std::min(bytes.size(), m_bytes.size());
    std::copy(bytes.begin(), bytes.begin() + bytes_to_copy, m_bytes.begin());
  }

  DynamicBitset() : m_size(0) {}

  /// Set the bit at the specified index to 1.
  ///
  /// \param[in] index
  ///     The zero-based index of the bit to set.
  void Set(size_t index) { m_bytes[index / 8] |= (1 << (index % 8)); }

  /// Clear the bit at the specified index (set to 0).
  ///
  /// \param[in] index
  ///     The zero-based index of the bit to clear.
  void Clear(size_t index) { m_bytes[index / 8] &= ~(1 << (index % 8)); }

  /// Get the value of the bit at the specified index.
  ///
  /// \param[in] index
  ///     The zero-based index of the bit to test.
  ///
  /// \return
  ///     True if the bit is set (1), false if clear (0).
  bool Get(size_t index) const {
    return m_bytes[index / 8] & (1 << (index % 8));
  }

  /// Array subscript operator for bit access.
  ///
  /// \param[in] index
  ///     The zero-based index of the bit to test.
  ///
  /// \return
  ///     True if the bit is set (1), false if clear (0).
  bool operator[](size_t index) const { return Get(index); }

  /// \return
  ///     A string containing the hexadecimal representation of the bitset.
  std::string AsHex() const;

private:
  /// The number of bits in the bitset.
  size_t m_size;

  /// The underlying byte storage for the bits.
  /// Each byte contains 8 bits, with bit 0 stored in the least significant bit.
  std::vector<uint8_t> m_bytes;
};

/// A simple bitset implementation wrapping an integral type used as a mask.
template <typename T,
          typename = typename std::enable_if<std::is_integral<T>::value>::type>
class StaticBitset {
public:
  StaticBitset(T storage) : m_storage(storage) {}

  /// Get the value of the bit at the specified index.
  ///
  /// \param[in] index
  ///     The zero-based index of the bit to test.
  ///
  /// \return
  ///     True if the bit is set (1), false if clear (0).
  bool Get(size_t index) const { return m_storage & (static_cast<T>(1) << index); }

  /// Array subscript operator for bit access.
  ///
  /// \param[in] index
  ///     The zero-based index of the bit to test.
  ///
  /// \return
  ///     True if the bit is set (1), false if clear (0).
  bool operator[](size_t index) const { return Get(index); }

  /// Update the bitset with a new storage value.
  ///
  /// \param[in] storage
  ///     The new storage value to update the bitset with.
  void Update(T storage) { m_storage = storage; }

  /// \return
  ///     True if all bits are set, false otherwise.
  bool AreAllBitsSet() const {
    return m_storage == std::numeric_limits<T>::max();
  }

  /// \return
  ///     The storage value.
  T GetStorage() const { return m_storage; }

private:
  T m_storage;
};
} // namespace lldb_private::lldb_server

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_UTILS_BITSET_H
