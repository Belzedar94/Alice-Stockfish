/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "engine.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cassert>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <new>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "alice_search.h"
#include "evaluate.h"
#include "misc.h"
#include "nnue/alice_native/alice_native_features.h"
#include "nnue/alice_native/alice_native_session.h"
#include "nnue/network.h"
#include "nnue/nnue_common.h"
#include "numa.h"
#include "perft.h"
#include "position.h"
#include "search.h"
#include "shm.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"

namespace Stockfish {

constexpr int MaxHashMB  = Is64Bit ? 33554432 : 2048;
int           MaxThreads = std::max(1024, 4 * int(get_hardware_concurrency()));

#ifdef ALICE_EVALFILE_DEFAULT
constexpr const char* DefaultLegacyEvalFile = ALICE_EVALFILE_DEFAULT;
#else
constexpr const char* DefaultLegacyEvalFile = "";
#endif

// The default configuration will attempt to group L3 domains up to 32 threads.
// This size was found to be a good balance between the Elo gain of increased
// history sharing and the speed loss from more cross-cache accesses (see
// PR#6526). The user can always explicitly override this behavior.
constexpr NumaAutoPolicy DefaultNumaPolicy = BundledL3Policy{32};

namespace {

enum class AliceEvaluationBackend : u8 {
    LEGACY,
    NATIVE,
    ZERO
};

AliceEvaluationBackend selected_evaluation_backend(const OptionsMap& options) {
    if (!bool(options["Use NNUE"]) || options["Alice Evaluation"] == "Zero")
        return AliceEvaluationBackend::ZERO;
    if (options["Alice Evaluation"] == "Native")
        return AliceEvaluationBackend::NATIVE;
    return AliceEvaluationBackend::LEGACY;
}

struct SearchPositionIdentity {
    const StateInfo* state      = nullptr;
    Key              key        = 0;
    Bitboard         boardB     = 0;
    Color            sideToMove = WHITE;
    int              pieceCount = 0;
};

SearchPositionIdentity search_position_identity(const Position& position) noexcept {
    return {position.state(), position.key(), position.state()->boardB, position.side_to_move(),
            position.count<ALL_PIECES>()};
}

bool same_search_position(const SearchPositionIdentity& identity,
                          const Position&               position) noexcept {
    return identity.state == position.state() && identity.key == position.key()
        && identity.boardB == position.state()->boardB
        && identity.sideToMove == position.side_to_move()
        && identity.pieceCount == position.count<ALL_PIECES>();
}

class SearchPositionStack {
   public:
    explicit SearchPositionStack(const Position& root) {
        frames[0] = search_position_identity(root);
    }

    bool matches(const Position& position, AliceSearch::EvalFailure& failure) const noexcept {
        if (same_search_position(frames[depth], position))
            return true;
        failure.code  = AliceSearch::EvalFailureCode::POSITION_MISMATCH;
        failure.stage = AliceSearch::EvalStage::EVALUATE;
        return false;
    }

    bool push(const Position& position, AliceSearch::EvalFailure& failure) noexcept {
        if (depth >= MAX_PLY)
        {
            failure.code  = AliceSearch::EvalFailureCode::STACK_OVERFLOW;
            failure.stage = AliceSearch::EvalStage::PUSH;
            return false;
        }
        if (position.state()->previous != frames[depth].state)
        {
            failure.code  = AliceSearch::EvalFailureCode::POSITION_MISMATCH;
            failure.stage = AliceSearch::EvalStage::PUSH;
            return false;
        }
        frames[++depth] = search_position_identity(position);
        return true;
    }

    bool pop(const Position& restoredParent, AliceSearch::EvalFailure& failure) noexcept {
        if (depth == 0)
        {
            failure.code  = AliceSearch::EvalFailureCode::STACK_UNDERFLOW;
            failure.stage = AliceSearch::EvalStage::POP;
            return false;
        }
        --depth;
        if (same_search_position(frames[depth], restoredParent))
            return true;
        failure.code  = AliceSearch::EvalFailureCode::POSITION_MISMATCH;
        failure.stage = AliceSearch::EvalStage::POP;
        return false;
    }

    usize size() const noexcept { return depth; }

   private:
    std::array<SearchPositionIdentity, MAX_PLY + 1> frames{};
    usize                                           depth = 0;
};

class LegacySearchEvaluator final: public AliceSearch::Evaluator {
   public:
    LegacySearchEvaluator(LegacyAliceExact&                              evaluator,
                          std::unique_ptr<LegacyAliceExact::Accumulator> accumulator,
                          const Position&                                root) :
        legacy(evaluator),
        state(std::move(accumulator)),
        positions(root) {}

    AliceSearch::EvaluatorIdentity identity() const noexcept override {
        return {"LegacyAliceExact", 0, legacy.metadata().sha256};
    }

    bool evaluate(const Position&           position,
                  Value&                    value,
                  AliceSearch::EvalFailure& failure) noexcept override {
        if (!state || !positions.matches(position, failure))
        {
            if (!failure)
            {
                failure.code  = AliceSearch::EvalFailureCode::NOT_READY;
                failure.stage = AliceSearch::EvalStage::EVALUATE;
            }
            return false;
        }
        const auto result = legacy.evaluate(position, *state, true);
        if (!result)
        {
            failure.code  = AliceSearch::EvalFailureCode::NOT_READY;
            failure.stage = AliceSearch::EvalStage::EVALUATE;
            return false;
        }
        value = *result;
        return true;
    }

    bool push(const Position&           position,
              const Dirties&            dirties,
              AliceSearch::EvalFailure& failure) noexcept override {
        if (!state)
        {
            failure.code  = AliceSearch::EvalFailureCode::NOT_READY;
            failure.stage = AliceSearch::EvalStage::PUSH;
            return false;
        }
        if (!positions.push(position, failure))
            return false;
        legacy.push(*state, position, dirties);
        return true;
    }

    bool pop(const Position& restoredParent, AliceSearch::EvalFailure& failure) noexcept override {
        if (!state)
        {
            failure.code  = AliceSearch::EvalFailureCode::NOT_READY;
            failure.stage = AliceSearch::EvalStage::POP;
            return false;
        }
        if (positions.size() == 0)
            return positions.pop(restoredParent, failure);
        legacy.pop(*state);
        return positions.pop(restoredParent, failure);
    }

   private:
    LegacyAliceExact&                              legacy;
    std::unique_ptr<LegacyAliceExact::Accumulator> state;
    SearchPositionStack                            positions;
};

class ZeroSearchEvaluator final: public AliceSearch::Evaluator {
   public:
    explicit ZeroSearchEvaluator(const Position& root) :
        positions(root) {}

    AliceSearch::EvaluatorIdentity identity() const noexcept override {
        return {"ZeroDiagnostic", 0, {}};
    }

    bool evaluate(const Position&           position,
                  Value&                    value,
                  AliceSearch::EvalFailure& failure) noexcept override {
        if (!positions.matches(position, failure))
            return false;
        value = VALUE_ZERO;
        return true;
    }

    bool push(const Position& position,
              const Dirties&,
              AliceSearch::EvalFailure& failure) noexcept override {
        return positions.push(position, failure);
    }

    bool pop(const Position& restoredParent, AliceSearch::EvalFailure& failure) noexcept override {
        return positions.pop(restoredParent, failure);
    }

   private:
    SearchPositionStack positions;
};

enum class ContractFailurePoint : u8 {
    NONE,
    EVALUATE,
    PUSH,
    POP
};

class ContractSearchEvaluator final: public AliceSearch::Evaluator {
   public:
    ContractSearchEvaluator(const Position& root, ContractFailurePoint requestedFailure) :
        positions(root),
        failurePoint(requestedFailure) {}

    AliceSearch::EvaluatorIdentity identity() const noexcept override {
        return {"ContractProbe", 0, {}};
    }

    bool evaluate(const Position&           position,
                  Value&                    value,
                  AliceSearch::EvalFailure& failure) noexcept override {
        ++evaluationCalls;
        if (!positions.matches(position, failure))
            return false;
        if (!failureInjected && failurePoint == ContractFailurePoint::EVALUATE)
        {
            failureInjected = true;
            failure.code    = AliceSearch::EvalFailureCode::INTERNAL_INVARIANT;
            failure.stage   = AliceSearch::EvalStage::EVALUATE;
            return false;
        }
        value = VALUE_ZERO;
        return true;
    }

    bool push(const Position& position,
              const Dirties&,
              AliceSearch::EvalFailure& failure) noexcept override {
        ++pushCalls;
        if (!failureInjected && failurePoint == ContractFailurePoint::PUSH)
        {
            failureInjected = true;
            failure.code    = AliceSearch::EvalFailureCode::INTERNAL_INVARIANT;
            failure.stage   = AliceSearch::EvalStage::PUSH;
            return false;
        }
        return positions.push(position, failure);
    }

    bool pop(const Position& restoredParent, AliceSearch::EvalFailure& failure) noexcept override {
        ++popCalls;
        if (!positions.pop(restoredParent, failure))
            return false;
        if (!failureInjected && failurePoint == ContractFailurePoint::POP)
        {
            failureInjected = true;
            failure.code    = AliceSearch::EvalFailureCode::INTERNAL_INVARIANT;
            failure.stage   = AliceSearch::EvalStage::POP;
            return false;
        }
        return true;
    }

    u64   evaluations() const noexcept { return evaluationCalls; }
    u64   pushes() const noexcept { return pushCalls; }
    u64   pops() const noexcept { return popCalls; }
    usize depth() const noexcept { return positions.size(); }

   private:
    SearchPositionStack  positions;
    ContractFailurePoint failurePoint;
    bool                 failureInjected = false;
    u64                  evaluationCalls = 0;
    u64                  pushCalls       = 0;
    u64                  popCalls        = 0;
};

struct ContractCaseResult {
    AliceSearch::Result result;
    u64                 evaluations = 0;
    u64                 pushes      = 0;
    u64                 pops        = 0;
    usize               depth       = 0;
    u64                 iterations  = 0;
    bool                rootMatches = false;
};

std::optional<std::string> run_search_contract_case(ContractFailurePoint point,
                                                    Depth                depth,
                                                    bool                 stopBeforeStart,
                                                    ContractCaseResult&  output) {
    StateInfo rootState;
    Position  position;
    if (auto error = position.set(StartFEN, false, &rootState))
        return error->what();

    const std::string rootFen = position.fen();
    const Key         rootKey = position.key();
    std::vector<Move> rootMoves;
    for (Move move : MoveList<LEGAL>(position))
        rootMoves.push_back(move);

    AliceSearch::Limits limits;
    limits.depth = depth;
    std::atomic_bool        stop(stopBeforeStart);
    ContractSearchEvaluator evaluator(position, point);
    output.result      = AliceSearch::search(position, rootMoves, limits, evaluator, stop,
                                             [&](const AliceSearch::Result&) { ++output.iterations; });
    output.evaluations = evaluator.evaluations();
    output.pushes      = evaluator.pushes();
    output.pops        = evaluator.pops();
    output.depth       = evaluator.depth();
    output.rootMatches = position.fen() == rootFen && position.key() == rootKey;
    return std::nullopt;
}

std::string format_search_failure(const AliceSearch::EvaluatorIdentity& identity,
                                  const AliceSearch::Result&            result) {
    std::ostringstream out;
    out << "Alice search evaluator failed backend=" << identity.backend
        << " code=" << AliceSearch::failure_code_name(result.failure.code)
        << " stage=" << AliceSearch::failure_stage_name(result.failure.stage)
        << " ply=" << result.failure.ply << " generation=" << identity.generation;
    if (!identity.sha256.empty())
        out << " sha256=" << identity.sha256;
    out << " root_restored=" << (result.rootRestored ? "yes" : "no");
    return out.str();
}

}  // namespace

Engine::Engine(std::optional<std::filesystem::path>) :
    numaContext(NumaConfig::from_system(DefaultNumaPolicy)),
    states(new std::deque<StateInfo>(1)),
    threads(),
    network(numaContext, std::make_unique<Eval::NNUE::Network>()) {

    pos.set(StartFEN, false, &states->back());

    options.add(  //
      "Debug Log File", Option("", [](const Option& o) {
          start_logger(path_from_utf8(std::string(o)));
          return std::nullopt;
      }));

    options.add(  //
      "NumaPolicy", Option("auto", [this](const Option& o) {
          if (!set_numa_config_from_option(o))
              return "NumaPolicy: invalid value '" + std::string(o) + "', keeping previous config.";
          return numa_config_information_as_string() + "\n"
               + thread_allocation_information_as_string();
      }));

    options.add(  //
      "Threads", Option(1, 1, MaxThreads, [this](const Option&) {
          resize_threads();
          return thread_allocation_information_as_string();
      }));

    options.add(  //
      "Hash", Option(16, 1, MaxHashMB, [this](const Option& o) {
          set_tt_size(o);
          return std::nullopt;
      }));

    options.add(  //
      "Clear Hash", Option([this](const Option&) {
          search_clear();
          return std::nullopt;
      }));

    options.add(  //
      "Ponder", Option(false));

    options.add(  //
      "MultiPV", Option(1, 1, MAX_MOVES));

    options.add("Skill Level", Option(20, 0, 20));

    options.add("Move Overhead", Option(10, 0, 5000));

    options.add("nodestime", Option(0, 0, 10000));

    options.add("UCI_Chess960", Option(false));

    options.add("UCI_LimitStrength", Option(false));

    options.add("UCI_Elo",
                Option(Stockfish::Search::Skill::LowestElo, Stockfish::Search::Skill::LowestElo,
                       Stockfish::Search::Skill::HighestElo));

    options.add("UCI_ShowWDL", Option(false));

    options.add(
      "Alice Evaluation", Option("Legacy var Native var Zero", "Legacy", [this](const Option& o) {
          if (o == "Native")
              return std::optional<std::string>(
                nativeQualification.loaded()
                  ? nativeQualification.status_line()
                  : "AliceNativeV1 selected; load an authenticated native EvalFile before eval or go.");
          if (o == "Zero")
              return std::optional<std::string>(
                "Deterministic zero diagnostic evaluation selected.");
          return std::optional<std::string>(
            legacyEvaluator.loaded()
              ? legacyEvaluator.status_line()
              : "LegacyAliceExact selected; load a compatible EvalFile before eval or go.");
      }));

    options.add(  //
      "Use NNUE", Option(true, [this](const Option& o) {
          if (!int(o))
              return std::optional<std::string>(
                "Use NNUE disabled; deterministic zero diagnostic evaluation overrides Alice Evaluation.");
          if (options["Alice Evaluation"] == "Native")
              return std::optional<std::string>(
                nativeQualification.loaded()
                  ? nativeQualification.status_line()
                  : "Use NNUE enabled with AliceNativeV1 selected; load authenticated native parameters before eval or go.");
          if (options["Alice Evaluation"] == "Zero")
              return std::optional<std::string>(
                "Use NNUE enabled; deterministic zero diagnostic evaluation remains selected.");
          return std::optional<std::string>(
            legacyEvaluator.loaded()
              ? legacyEvaluator.status_line()
              : "Use NNUE enabled with LegacyAliceExact selected; load a compatible EvalFile before eval or go.");
      }));

    options.add(  //
      "Alice Native EvalFile",
      Option("", [this](const Option&) { return configure_native_network(); }));

    options.add(  //
      "Alice Native SHA256",
      Option("", [this](const Option&) { return configure_native_network(); }));

    options.add(  //
      "Alice_Frozen_Network", Option(true, [this](const Option&) {
          const auto file = path_from_utf8(std::string(options["EvalFile"]));
          if (file.empty())
              return std::optional<std::string>(
                "Frozen-network policy updated; no EvalFile is selected.");
          return configure_legacy_network(file);
      }));

    options.add(  //
      "EvalFile", Option(DefaultLegacyEvalFile, [this](const Option& o) {
          return configure_legacy_network(path_from_utf8(std::string(o)));
      }));

    threads.clear();
    threads.ensure_network_replicated();
    resize_threads();

    if (DefaultLegacyEvalFile[0] != '\0')
        configure_legacy_network(path_from_utf8(DefaultLegacyEvalFile));
}

Engine::~Engine() {
    stop();
    wait_for_search_finished();
}

std::variant<u64, PositionSetError>
Engine::perft(const std::string& fen, Depth depth, bool isChess960) {
    wait_for_search_finished();
    return Benchmark::perft(fen, depth, isChess960);
}

std::optional<std::string> Engine::go(Search::LimitsType& limits) {
    assert(limits.perft == 0);

    wait_for_search_finished();

    const AliceEvaluationBackend evaluationBackend = selected_evaluation_backend(options);
    if (evaluationBackend == AliceEvaluationBackend::LEGACY && !legacyEvaluator.loaded())
        return "Legacy Alice evaluation is enabled, but no compatible network is loaded"
             + (legacyEvaluator.last_error().empty() ? std::string(".")
                                                     : ": " + legacyEvaluator.last_error());

    if (evaluationBackend == AliceEvaluationBackend::LEGACY && !pos.is_draw(0))
    {
        verify_network();
        threads.start_thinking(options, pos, states, limits);
        return std::nullopt;
    }

    std::unique_ptr<LegacyAliceExact::Accumulator> legacyAccumulator;
    if (evaluationBackend == AliceEvaluationBackend::LEGACY)
    {
        legacyAccumulator = legacyEvaluator.make_accumulator(pos);
        if (!legacyAccumulator)
            return "Legacy Alice evaluation could not create its root accumulator.";
    }

    std::optional<Eval::NNUE::AliceNative::QualificationNetwork::Lease> nativeLease;
    if (evaluationBackend == AliceEvaluationBackend::NATIVE)
    {
        std::string leaseError;
        nativeLease = lease_native_network(leaseError);
        if (!nativeLease)
            return "AliceNativeV1 is selected, but its parameters cannot be leased: " + leaseError;
    }

    verify_network();
    aliceSearchStop.store(false, std::memory_order_relaxed);
    alicePondering.store(limits.ponderMode, std::memory_order_relaxed);
    threads.stop = false;

    std::vector<Move> legalMoves;
    for (Move move : MoveList<LEGAL>(pos))
        legalMoves.push_back(move);

    std::vector<Move> rootMoves;
    for (const std::string& moveText : limits.searchmoves)
    {
        const Move move = UCIEngine::to_move(pos, moveText);
        if (move != Move::none()
            && std::find(rootMoves.begin(), rootMoves.end(), move) == rootMoves.end())
            rootMoves.push_back(move);
    }

    // Match the established UCI behavior: an empty or wholly invalid
    // searchmoves list falls back to the complete legal root set.
    if (rootMoves.empty())
        rootMoves = legalMoves;

    AliceSearch::Limits aliceLimits;
    if (limits.depth > 0)
        aliceLimits.depth = std::clamp(limits.depth, 1, MAX_PLY);
    else if (limits.mate > 0)
        aliceLimits.depth = std::clamp(2 * limits.mate, 1, MAX_PLY);
    else if (limits.infinite || limits.ponderMode || limits.nodes || limits.movetime
             || limits.use_time_management())
        aliceLimits.depth = MAX_PLY;
    else
        aliceLimits.depth = 5;

    aliceLimits.nodes = limits.nodes;

    if (!limits.ponderMode)
    {
        const TimePoint overhead = TimePoint(options["Move Overhead"]);
        TimePoint       budget   = 0;

        if (limits.movetime > 0)
            budget = std::max(TimePoint(1), limits.movetime - overhead);
        else if (limits.use_time_management())
        {
            const Color     us        = pos.side_to_move();
            const TimePoint remaining = limits.time[us];
            const int       moves     = limits.movestogo > 0 ? limits.movestogo : 30;
            const TimePoint share     = remaining / moves + 3 * limits.inc[us] / 4;
            budget = std::clamp(share - overhead, TimePoint(1), std::max(TimePoint(1), remaining));
        }

        if (budget > 0)
            aliceLimits.deadline = now() + budget;
    }

    assert(states);
    const std::string rootFen     = pos.fen();
    const StateInfo   rootState   = states->back();
    const bool        isChess960  = pos.is_chess960();
    const bool        waitForStop = limits.infinite;

    aliceSearchThread = std::thread([this, rootMoves = std::move(rootMoves), aliceLimits, rootFen,
                                     rootState, isChess960, waitForStop, evaluationBackend,
                                     legacyAccumulator = std::move(legacyAccumulator),
                                     nativeLease       = std::move(nativeLease)]() mutable {
        StateInfo  searchRootState;
        Position   searchPos;
        const auto error = searchPos.set(rootFen, isChess960, &searchRootState);
        assert(!error.has_value());
        (void) error;
        searchRootState = rootState;

        std::unique_ptr<AliceSearch::Evaluator> evaluator;
        if (evaluationBackend == AliceEvaluationBackend::LEGACY)
            evaluator = std::make_unique<LegacySearchEvaluator>(
              legacyEvaluator, std::move(legacyAccumulator), searchPos);
        else if (evaluationBackend == AliceEvaluationBackend::NATIVE)
        {
            assert(nativeLease.has_value());
            std::unique_ptr<Eval::NNUE::AliceNative::SearchSession> nativeSession(
              new (std::nothrow) Eval::NNUE::AliceNative::SearchSession(
                nativeLease->parameter_view(), nativeLease->generation(), nativeLease->sha256(),
                searchPos));
            if (!nativeSession)
            {
                if (onSearchError)
                {
                    AliceSearch::Result failed;
                    failed.completion         = AliceSearch::Completion::FAILED;
                    failed.failure.code       = AliceSearch::EvalFailureCode::NOT_READY;
                    failed.failure.stage      = AliceSearch::EvalStage::ROOT_REFRESH;
                    failed.failure.generation = nativeLease->generation();
                    failed.rootRestored       = true;
                    onSearchError(format_search_failure(
                      {"AliceNativeV1", nativeLease->generation(), nativeLease->sha256()}, failed));
                }
                return;
            }
            if (!nativeSession->ready())
            {
                Value                    ignored = VALUE_ZERO;
                AliceSearch::EvalFailure failure;
                nativeSession->evaluate(searchPos, ignored, failure);
                AliceSearch::Result failed;
                failed.completion   = AliceSearch::Completion::FAILED;
                failed.failure      = failure;
                failed.rootRestored = true;
                if (onSearchError)
                    onSearchError(format_search_failure(nativeSession->identity(), failed));
                return;
            }
            evaluator = std::move(nativeSession);
        }
        else
            evaluator = std::make_unique<ZeroSearchEvaluator>(searchPos);

        const TimePoint started = now();

        if (updateContext.onStart)
            updateContext.onStart();

        const auto onIteration = [this, started, &searchPos](const AliceSearch::Result& result) {
            std::string pv;
            for (Move move : result.pv)
            {
                if (!pv.empty())
                    pv += ' ';
                pv += UCIEngine::move(move, searchPos.is_chess960());
            }

            const std::string wdl     = result.score > VALUE_DRAW ? "1000 0 0"
                                      : result.score < VALUE_DRAW ? "0 0 1000"
                                                                  : "0 1000 0";
            const TimePoint   elapsed = std::max(TimePoint(1), now() - started);

            InfoFull info;
            info.depth    = result.depth;
            info.selDepth = result.depth;
            info.multiPV  = 1;
            info.score    = Score(result.score, searchPos);
            info.wdl      = wdl;
            info.bound    = "";
            info.timeMs   = usize(elapsed);
            info.nodes    = usize(result.nodes);
            info.nps      = usize(result.nodes * 1000 / u64(elapsed));
            info.tbHits   = 0;
            info.pv       = pv;
            info.hashfull = 0;

            if (updateContext.onUpdateFull)
                updateContext.onUpdateFull(info);
        };

        const AliceSearch::Result result = AliceSearch::search(
          searchPos, rootMoves, aliceLimits, *evaluator, aliceSearchStop, onIteration);

        if (result.completion == AliceSearch::Completion::FAILED)
        {
            if (onSearchError)
                onSearchError(format_search_failure(evaluator->identity(), result));
            return;
        }

        if (rootMoves.empty() && updateContext.onUpdateNoMoves)
            updateContext.onUpdateNoMoves({0, Score(result.score, searchPos)});

        if (result.terminal != AliceSearch::Terminal::NONE)
        {
            std::string_view gameResult = "1/2-1/2";
            std::string_view reason     = "rule_draw";
            if (result.terminal == AliceSearch::Terminal::CHECKMATE)
            {
                gameResult = searchPos.side_to_move() == WHITE ? "0-1" : "1-0";
                reason     = "checkmate";
            }
            else if (result.terminal == AliceSearch::Terminal::STALEMATE)
                reason = "stalemate";

            sync_cout << "info string alice_result result=" << gameResult << " reason=" << reason
                      << sync_endl;
        }

        while (!aliceSearchStop.load(std::memory_order_relaxed)
               && (waitForStop || alicePondering.load(std::memory_order_relaxed)))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        std::string bestmove = UCIEngine::move(result.bestMove, searchPos.is_chess960());
        std::string ponder;
        if (result.pv.size() > 1)
            ponder = UCIEngine::move(result.pv[1], searchPos.is_chess960());

        if (updateContext.onBestmove)
            updateContext.onBestmove(bestmove, ponder);
    });
    return std::nullopt;
}
void Engine::stop() {
    aliceSearchStop.store(true, std::memory_order_relaxed);
    alicePondering.store(false, std::memory_order_relaxed);
    threads.stop = true;
}

void Engine::search_clear() {
    wait_for_search_finished();

    tt.clear(threads);
    threads.clear();
}

void Engine::set_on_update_no_moves(std::function<void(const Engine::InfoShort&)>&& f) {
    updateContext.onUpdateNoMoves = std::move(f);
}

void Engine::set_on_update_full(std::function<void(const Engine::InfoFull&)>&& f) {
    updateContext.onUpdateFull = std::move(f);
}

void Engine::set_on_iter(std::function<void(const Engine::InfoIter&)>&& f) {
    updateContext.onIter = std::move(f);
}

void Engine::set_on_bestmove(std::function<void(std::string_view, std::string_view)>&& f) {
    updateContext.onBestmove = std::move(f);
}

void Engine::set_on_start(std::function<void()>&& f) { updateContext.onStart = std::move(f); }

void Engine::set_on_search_error(std::function<void(std::string_view)>&& f) {
    onSearchError = std::move(f);
}

void Engine::set_on_verify_network(std::function<void(std::string_view)>&& f) {
    onVerifyNetwork = std::move(f);
}

void Engine::wait_for_search_finished() {
    if (aliceSearchThread.joinable())
        aliceSearchThread.join();
    threads.main_thread()->wait_for_search_finished();
}

std::optional<PositionSetError> Engine::set_position(const std::string&              fen,
                                                     const std::vector<std::string>& moves) {
    wait_for_search_finished();

    // Validate the complete command on an isolated position. A bad FEN or a bad
    // move therefore leaves the current game and every StateInfo pointer intact.
    Position     candidate;
    StateListPtr candidateStates(new std::deque<StateInfo>(1));
    auto         err = candidate.set(fen, options["UCI_Chess960"], &candidateStates->back());
    if (err.has_value())
        return err;

    std::vector<Move> resolvedMoves;
    resolvedMoves.reserve(moves.size());
    for (const auto& move : moves)
    {
        const Move resolved = UCIEngine::to_move(candidate, move);

        if (resolved == Move::none())
            return PositionSetError("Illegal move: " + move);

        resolvedMoves.push_back(resolved);
        candidateStates->emplace_back();
        candidate.do_move(resolved, candidateStates->back());
    }

    StateListPtr newStates(new std::deque<StateInfo>(1));
    err = pos.set(fen, options["UCI_Chess960"], &newStates->back());
    assert(!err.has_value());
    for (Move move : resolvedMoves)
    {
        newStates->emplace_back();
        pos.do_move(move, newStates->back());
    }
    states = std::move(newStates);

    return std::nullopt;
}

// modifiers

bool Engine::set_numa_config_from_option(const std::string& o) {
    if (o == "auto" || o == "system")
    {
        numaContext.set_numa_config(NumaConfig::from_system(DefaultNumaPolicy));
    }
    else if (o == "hardware")
    {
        // Don't respect affinity set in the system.
        numaContext.set_numa_config(NumaConfig::from_system(DefaultNumaPolicy, false));
    }
    else if (o == "none")
    {
        numaContext.set_numa_config(NumaConfig{});
    }
    else
    {
        auto parsed = NumaConfig::from_string(o);
        if (!parsed.has_value())
            return false;
        numaContext.set_numa_config(std::move(*parsed));
    }

    // Force reallocation of threads in case affinities need to change.
    resize_threads();
    threads.ensure_network_replicated();
    return true;
}

void Engine::resize_threads() {
    threads.wait_for_search_finished();
    threads.set(numaContext.get_numa_config(),
                {options, threads, tt, sharedHists, network, legacyEvaluator}, updateContext);

    // Reallocate the hash with the new threadpool size
    set_tt_size(options["Hash"]);
    threads.ensure_network_replicated();
}

void Engine::set_tt_size(usize mb) {
    wait_for_search_finished();
    tt.resize(mb, threads);
}

void Engine::set_ponderhit(bool b) {
    if (!b && alicePondering.exchange(false, std::memory_order_relaxed))
        aliceSearchStop.store(true, std::memory_order_relaxed);
    threads.main_manager()->ponder = b;
}

// network related

void Engine::verify_network() const {
    if (!onVerifyNetwork)
        return;

    switch (selected_evaluation_backend(options))
    {
    case AliceEvaluationBackend::LEGACY :
        onVerifyNetwork(legacyEvaluator.status_line());
        break;
    case AliceEvaluationBackend::NATIVE :
        onVerifyNetwork(nativeQualification.status_line());
        break;
    case AliceEvaluationBackend::ZERO :
        onVerifyNetwork("Deterministic zero diagnostic evaluation is active.");
        break;
    }
}

// utility functions

std::optional<std::string> Engine::trace_eval() const {
    const AliceEvaluationBackend evaluationBackend = selected_evaluation_backend(options);
    if (evaluationBackend == AliceEvaluationBackend::ZERO)
    {
        sync_cout << "info string Deterministic zero diagnostic evaluation is active.\n"
                  << "legacy_nnue raw 0 adjusted 0" << sync_endl;
        return std::nullopt;
    }
    if (evaluationBackend == AliceEvaluationBackend::NATIVE)
    {
        std::string leaseError;
        auto        lease = lease_native_network(leaseError);
        if (!lease)
            return "AliceNativeV1 evaluation requires leased parameters: " + leaseError;

        std::unique_ptr<Eval::NNUE::AliceNative::SearchSession> session(
          new (std::nothrow) Eval::NNUE::AliceNative::SearchSession(
            lease->parameter_view(), lease->generation(), lease->sha256(), pos));
        if (!session)
            return "AliceNativeV1 evaluation could not allocate its fixed frame stack.";

        Value                    value = VALUE_ZERO;
        AliceSearch::EvalFailure failure;
        if (!session->evaluate(pos, value, failure))
        {
            AliceSearch::Result failed;
            failed.completion   = AliceSearch::Completion::FAILED;
            failed.failure      = failure;
            failed.rootRestored = session->matches_current(pos);
            return format_search_failure(session->identity(), failed);
        }

        sync_cout << "info string AliceNativeV1 loaded generation=" << lease->generation()
                  << " sha256=" << lease->sha256() << "\n"
                  << "alice_native value " << value << " generation " << lease->generation()
                  << " sha256 " << lease->sha256() << sync_endl;
        return std::nullopt;
    }

    if (!legacyEvaluator.loaded())
        return "Legacy Alice evaluation is enabled, but no compatible network is loaded"
             + (legacyEvaluator.last_error().empty() ? std::string(".")
                                                     : ": " + legacyEvaluator.last_error());

    const auto raw      = legacyEvaluator.evaluate(pos, false);
    const auto adjusted = legacyEvaluator.evaluate(pos, true);
    if (!raw || !adjusted)
        return "Legacy Alice evaluator became unavailable.";

    sync_cout << "info string " << legacyEvaluator.status_line() << "\n"
              << "legacy_nnue raw " << *raw << " adjusted " << *adjusted << sync_endl;
    return std::nullopt;
}

std::optional<std::string> Engine::verify_search_contract(std::string& report) {
    wait_for_search_finished();

    ContractCaseResult balanced;
    if (auto error = run_search_contract_case(ContractFailurePoint::NONE, 2, false, balanced))
        return "Alice search balanced contract case failed to start: " + *error;
    if (balanced.result.completion != AliceSearch::Completion::COMPLETED
        || !balanced.result.rootRestored || !balanced.rootMatches || balanced.depth != 0
        || balanced.pushes == 0 || balanced.pushes != balanced.pops || balanced.evaluations == 0
        || balanced.iterations != 2)
        return "Alice search balanced contract case did not preserve its stack and root.";

    auto verify_failure = [&](ContractFailurePoint point, AliceSearch::EvalStage expectedStage,
                              u64 expectedEvaluations, u64 expectedPushes,
                              u64 expectedPops) -> std::optional<std::string> {
        ContractCaseResult result;
        if (auto error = run_search_contract_case(point, 1, false, result))
            return error;
        if (result.result.completion != AliceSearch::Completion::FAILED
            || result.result.failure.code != AliceSearch::EvalFailureCode::INTERNAL_INVARIANT
            || result.result.failure.stage != expectedStage || !result.result.rootRestored
            || !result.rootMatches || result.depth != 0 || result.iterations != 0
            || result.evaluations != expectedEvaluations || result.pushes != expectedPushes
            || result.pops != expectedPops)
            return "injected " + std::string(AliceSearch::failure_stage_name(expectedStage))
                 + " failure did not unwind exactly";
        return std::nullopt;
    };

    if (auto error =
          verify_failure(ContractFailurePoint::EVALUATE, AliceSearch::EvalStage::EVALUATE, 1, 1, 1))
        return "Alice search contract rejected: " + *error + ".";
    if (auto error =
          verify_failure(ContractFailurePoint::PUSH, AliceSearch::EvalStage::PUSH, 0, 1, 0))
        return "Alice search contract rejected: " + *error + ".";
    if (auto error =
          verify_failure(ContractFailurePoint::POP, AliceSearch::EvalStage::POP, 1, 1, 1))
        return "Alice search contract rejected: " + *error + ".";

    ContractCaseResult stopped;
    if (auto error = run_search_contract_case(ContractFailurePoint::NONE, 2, true, stopped))
        return "Alice search stopped contract case failed to start: " + *error;
    if (stopped.result.completion != AliceSearch::Completion::STOPPED
        || !stopped.result.rootRestored || !stopped.rootMatches || stopped.depth != 0
        || stopped.evaluations != 0 || stopped.pushes != 0 || stopped.pops != 0
        || stopped.iterations != 0)
        return "Alice search stopped contract case was not distinct from failure.";

    std::ostringstream out;
    out << "alice_search contract verified cases 5 balanced_pushes " << balanced.pushes
        << " balanced_pops " << balanced.pops << " balanced_evaluations " << balanced.evaluations
        << " injected_failures 3 stopped_cases 1 root_restorations 5";
    report = out.str();
    return std::nullopt;
}

std::string Engine::trace_native_features() {
    wait_for_search_finished();
    return Eval::NNUE::AliceNative::trace_json(pos);
}

std::optional<std::string> Engine::verify_native_incremental(Depth depth, std::string& report) {
    wait_for_search_finished();
    Eval::NNUE::AliceNative::IncrementalVerificationStats stats;
    if (auto error = Eval::NNUE::AliceNative::verify_incremental(pos, depth, stats))
        return error;

    std::ostringstream out;
    out << "alice_native incremental verified positions " << stats.positions << " transitions "
        << stats.transitions << " captures " << stats.captures << " promotions " << stats.promotions
        << " castlings " << stats.castlings << " king_moves " << stats.kingMoves << " refreshes "
        << stats.fullRefreshes[WHITE] << ',' << stats.fullRefreshes[BLACK] << " max_piece_events "
        << stats.maxPieceEvents << " max_threat_events " << stats.maxThreatEvents
        << " cache_checks " << stats.cacheChecks << " cache_adds " << stats.cachePieceAdds
        << " cache_removes " << stats.cachePieceRemoves << " cache_board_b_events "
        << stats.cacheBoardBEvents << " simd_checks " << stats.simdChecks
        << " fixed_snapshot_checks " << stats.fixedSnapshotChecks << " depth " << depth;
    report = out.str();
    return std::nullopt;
}

std::optional<std::string>
Engine::validate_native_wire(const std::filesystem::path&      file,
                             const std::optional<std::string>& expectedSha256) {
    wait_for_search_finished();
    if (auto error = nativeWireValidator.validate(file, expectedSha256))
        return "Alice native wire rejected: " + *error;
    return std::nullopt;
}

std::string Engine::native_wire_status() const { return nativeWireValidator.status_line(); }

std::optional<std::string> Engine::load_native_qualification(const std::filesystem::path& file,
                                                             std::string_view expectedSha256) {
    if (auto error = nativeQualification.load(file, expectedSha256))
        return "Alice native qualification load rejected: " + *error;
    return std::nullopt;
}

std::string Engine::native_qualification_status() const {
    return nativeQualification.status_line();
}

std::string Engine::native_tensor_status() const {
    return nativeQualification.tensor_status_line();
}

std::optional<std::string>
Engine::probe_native_parameter(std::string_view tensor, u64 index, std::string& report) const {
    return nativeQualification.probe(tensor, index, report);
}

std::optional<std::string> Engine::trace_native_integer(std::string& report) {
    wait_for_search_finished();
    return nativeQualification.integer_trace(pos, report);
}

std::optional<std::string> Engine::verify_loaded_native_incremental(Depth        depth,
                                                                    std::string& report) {
    wait_for_search_finished();
    Eval::NNUE::AliceNative::LoadedIncrementalVerificationStats stats;
    if (auto error = nativeQualification.verify_incremental(pos, depth, stats))
        return error;

    std::ostringstream out;
    out << "alice_native loaded incremental verified generation "
        << nativeQualification.generation() << " positions " << stats.positions << " transitions "
        << stats.transitions << " captures " << stats.captures << " promotions " << stats.promotions
        << " castlings " << stats.castlings << " king_moves " << stats.kingMoves << " refreshes "
        << stats.fullRefreshes[WHITE] << ',' << stats.fullRefreshes[BLACK] << " piece_adds "
        << stats.pieceAdds << " piece_removes " << stats.pieceRemoves << " threat_adds "
        << stats.threatAdds << " threat_removes " << stats.threatRemoves << " max_piece_events "
        << stats.maxPieceEvents << " max_threat_events " << stats.maxThreatEvents
        << " accumulator_comparisons " << stats.accumulatorComparisons
        << " integer_stage_comparisons " << stats.integerStageComparisons
        << " feature_simd_comparisons " << stats.featureSimdComparisons
        << " dense_simd_comparisons " << stats.denseSimdComparisons << " fixed_accumulator_checks "
        << stats.fixedAccumulatorChecks << " fixed_delta_updates " << stats.fixedDeltaUpdates
        << " undo_checks " << stats.undoChecks << " depth " << depth << " search available";
    report = out.str();
    return std::nullopt;
}

std::optional<std::string> Engine::verify_native_search_session(Depth depth, std::string& report) {
    wait_for_search_finished();
    return nativeQualification.verify_session(pos, depth, report);
}

std::optional<std::string> Engine::verify_native_lease(std::string& report) {
    wait_for_search_finished();
    return nativeQualification.verify_lease_contract(report);
}

std::optional<std::string> Engine::verify_legacy_incremental(Depth depth, u64& positions) {
    wait_for_search_finished();
    positions = 0;

    if (!legacyEvaluator.loaded())
        return "Legacy Alice incremental verification requires a compatible loaded network.";
    if (depth < 0 || depth > 4)
        return "Legacy Alice incremental verification depth must be between 0 and 4.";

    auto accumulator = legacyEvaluator.make_accumulator(pos);
    if (!accumulator)
        return "Legacy Alice incremental accumulator initialization failed.";

    const std::string                                rootFen = pos.fen();
    const Key                                        rootKey = pos.key();
    std::function<std::optional<std::string>(Depth)> visit;
    visit = [&](Depth remaining) -> std::optional<std::string> {
        const auto fullRaw             = legacyEvaluator.evaluate(pos, false);
        const auto fullAdjusted        = legacyEvaluator.evaluate(pos, true);
        const auto incrementalRaw      = legacyEvaluator.evaluate(pos, *accumulator, false);
        const auto incrementalAdjusted = legacyEvaluator.evaluate(pos, *accumulator, true);

        if (!fullRaw || !fullAdjusted || !incrementalRaw || !incrementalAdjusted)
            return "Legacy Alice evaluator became unavailable during incremental verification.";
        if (*fullRaw != *incrementalRaw || *fullAdjusted != *incrementalAdjusted)
        {
            std::ostringstream error;
            error << "Legacy Alice incremental mismatch at " << pos.fen()
                  << ": full raw=" << *fullRaw << " adjusted=" << *fullAdjusted
                  << ", incremental raw=" << *incrementalRaw << " adjusted=" << *incrementalAdjusted
                  << ".";
            return error.str();
        }

        ++positions;
        if (remaining == 0)
            return std::nullopt;

        std::vector<Move> moves;
        for (Move move : MoveList<LEGAL>(pos))
            moves.push_back(move);

        for (Move move : moves)
        {
            StateInfo state;
            Dirties   dirties;
            pos.do_move(move, state, pos.gives_check(move), dirties, nullptr, nullptr);
            legacyEvaluator.push(*accumulator, pos, dirties);
            auto error = visit(remaining - 1);
            legacyEvaluator.pop(*accumulator);
            pos.undo_move(move);
            if (error)
                return error;
        }
        return std::nullopt;
    };

    auto error = visit(depth);
    if (pos.fen() != rootFen || pos.key() != rootKey)
        return "Legacy Alice incremental verification did not restore the root position.";
    return error;
}

std::optional<std::string> Engine::configure_legacy_network(const std::filesystem::path& file) {
    wait_for_search_finished();
    if (file.empty())
    {
        legacyEvaluator.reset();
        return "LegacyAliceExact unloaded; EvalFile is empty.";
    }

    const auto policy = bool(options["Alice_Frozen_Network"])
                        ? LegacyAliceExact::LoadPolicy::FrozenBaseline
                        : LegacyAliceExact::LoadPolicy::FormatCompatible;
    if (auto error = legacyEvaluator.load(file, policy))
        return "LegacyAliceExact rejected EvalFile: " + *error;
    return legacyEvaluator.status_line();
}

std::optional<std::string> Engine::configure_native_network() {
    const std::string file   = options["Alice Native EvalFile"];
    const std::string sha256 = options["Alice Native SHA256"];
    if (file.empty() && sha256.empty())
        return nativeQualification.loaded()
               ? std::optional<std::string>(
                   "Alice native EvalFile options are empty; the explicitly loaded parameters remain installed.")
               : std::optional<std::string>(
                   "Alice native EvalFile options are empty; no parameters are loaded.");
    if (file.empty() || sha256.empty())
        return "AliceNativeV1 loading requires both Alice Native EvalFile and Alice Native SHA256.";

    if (auto error = nativeQualification.load(path_from_utf8(file), sha256))
        return "AliceNativeV1 rejected EvalFile: " + *error;
    return nativeQualification.status_line();
}

std::optional<Eval::NNUE::AliceNative::QualificationNetwork::Lease>
Engine::lease_native_network(std::string& error) const {
    auto lease = nativeQualification.acquire_lease(error);
    if (!lease)
        return std::nullopt;

    const std::string selectedFile = options["Alice Native EvalFile"];
    const std::string selectedSha  = options["Alice Native SHA256"];
    if (selectedFile.empty() != selectedSha.empty())
    {
        error =
          "both Alice Native EvalFile and Alice Native SHA256 are required, or neither when parameters were loaded explicitly";
        return std::nullopt;
    }

    if (!selectedSha.empty())
    {
        const CaseInsensitiveLess less;
        const std::string         leasedSha(lease->sha256());
        if (less(selectedSha, leasedSha) || less(leasedSha, selectedSha))
        {
            error = "the selected SHA-256 does not match the installed parameters";
            return std::nullopt;
        }

        std::error_code pathError;
        const bool      samePath = std::filesystem::equivalent(
          path_from_utf8(selectedFile), path_from_utf8(std::string(lease->normalized_path())),
          pathError);
        if (pathError || !samePath)
        {
            error =
              "the selected EvalFile is unavailable or does not match the installed parameter source";
            return std::nullopt;
        }
    }

    error.clear();
    return lease;
}

const OptionsMap& Engine::get_options() const { return options; }
OptionsMap&       Engine::get_options() { return options; }

std::string Engine::fen() const { return pos.fen(); }

std::optional<PositionSetError> Engine::flip() {
    wait_for_search_finished();
    return pos.flip();
}

std::string Engine::visualize() const {
    std::stringstream ss;
    ss << pos;
    return ss.str();
}

int Engine::get_hashfull(int maxAge) const { return tt.hashfull(maxAge); }

std::vector<std::pair<usize, usize>> Engine::get_bound_thread_count_by_numa_node() const {
    auto                                 counts = threads.get_bound_thread_count_by_numa_node();
    const NumaConfig&                    cfg    = numaContext.get_numa_config();
    std::vector<std::pair<usize, usize>> ratios;
    NumaIndex                            n = 0;
    for (; n < counts.size(); ++n)
        ratios.emplace_back(counts[n], cfg.num_cpus_in_numa_node(n));
    if (!counts.empty())
        for (; n < cfg.num_numa_nodes(); ++n)
            ratios.emplace_back(0, cfg.num_cpus_in_numa_node(n));
    return ratios;
}

std::string Engine::get_numa_config_as_string() const {
    return numaContext.get_numa_config().to_string();
}

std::string Engine::numa_config_information_as_string() const {
    auto cfgStr = get_numa_config_as_string();
    return "Available processors: " + cfgStr;
}

std::string Engine::thread_binding_information_as_string() const {
    auto              boundThreadsByNode = get_bound_thread_count_by_numa_node();
    std::stringstream ss;
    if (boundThreadsByNode.empty())
        return ss.str();

    bool isFirst = true;

    for (auto&& [current, total] : boundThreadsByNode)
    {
        if (!isFirst)
            ss << ":";
        ss << current << "/" << total;
        isFirst = false;
    }

    return ss.str();
}

std::string Engine::thread_allocation_information_as_string() const {
    std::stringstream ss;

    usize threadsSize = threads.size();
    ss << "Using " << threadsSize << (threadsSize > 1 ? " threads" : " thread");

    auto boundThreadsByNodeStr = thread_binding_information_as_string();
    if (boundThreadsByNodeStr.empty())
        return ss.str();

    ss << " with NUMA node thread binding: ";
    ss << boundThreadsByNodeStr;

    return ss.str();
}
}
