/*
  Alice-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Alice-Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.
*/

#ifndef LEGACY_ALICE_NNUE_H_INCLUDED
#define LEGACY_ALICE_NNUE_H_INCLUDED

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "types.h"

namespace Stockfish {

class Position;

class LegacyAliceExact {
   public:
    enum class LoadPolicy {
        FrozenBaseline,
        FormatCompatible
    };

    struct Metadata {
        std::string normalizedPath;
        std::string sha256;
        std::string description;
        u32         version      = 0;
        u32         architecture = 0;
        u64         fileSize     = 0;
        bool        frozen       = false;
    };

    LegacyAliceExact();
    ~LegacyAliceExact();

    LegacyAliceExact(const LegacyAliceExact&)            = delete;
    LegacyAliceExact(LegacyAliceExact&&)                 = delete;
    LegacyAliceExact& operator=(const LegacyAliceExact&) = delete;
    LegacyAliceExact& operator=(LegacyAliceExact&&)      = delete;

    std::optional<std::string> load(const std::filesystem::path&, LoadPolicy);
    void                       reset();

    bool                 loaded() const;
    std::optional<Value> evaluate(const Position&, bool adjusted) const;
    const Metadata&      metadata() const;
    const std::string&   last_error() const;
    std::string          status_line() const;

    static constexpr const char* RequiredFileName = "alice_run2rl_e40_l09.nnue";
    static constexpr const char* FrozenSha256 =
      "9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9";
    static constexpr u32 ExpectedVersion      = 0x7AF32F20U;
    static constexpr u32 ExpectedArchitecture = 0x3C103E72U;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Stockfish

#endif  // LEGACY_ALICE_NNUE_H_INCLUDED
