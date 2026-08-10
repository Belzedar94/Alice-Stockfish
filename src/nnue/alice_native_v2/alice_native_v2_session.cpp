/*
  Alice-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "alice_native_v2_session.h"

#include <algorithm>

#include "../../position.h"

namespace Stockfish::Eval::NNUE::AliceNativeV2 {

namespace {

Board board_at(Bitboard boardB, Square square) noexcept {
    return boardB & square ? BOARD_B : BOARD_A;
}

}  // namespace

SearchSession::SearchSession(const ParameterView& loadedParameters, u64 generation,
                             std::string_view sha256, const Position& root) noexcept :
    parameters(loadedParameters), parameterGeneration(generation), parameterSha256(sha256) {
    initialized = initialize(root);
}

AliceSearch::EvaluatorIdentity SearchSession::identity() const noexcept {
    return {"AliceNativeV2M512", parameterGeneration, parameterSha256};
}

SearchSession::PositionIdentity SearchSession::capture(const Position& position) noexcept {
    return {position.state(), position.key(), position.state()->boardB, position.side_to_move(),
            position.count<ALL_PIECES>()};
}

bool SearchSession::same_position(const PositionIdentity& expected,
                                  const Position& position) noexcept {
    return expected.state == position.state() && expected.key == position.key()
        && expected.boardB == position.state()->boardB
        && expected.sideToMove == position.side_to_move()
        && expected.pieceCount == position.count<ALL_PIECES>();
}

bool SearchSession::complete(const ParameterView& view) noexcept {
    if (!view.ftBias || !view.pieceSquareWeight || !view.pieceSquarePsqt)
        return false;
    for (const auto& dense : view.dense)
        if (!dense.fc0Bias || !dense.fc0Weight || !dense.fc0InterleavedWeight || !dense.fc1Bias
            || !dense.fc1Weight
            || !dense.fc2Bias || !dense.fc2Weight)
            return false;
    return true;
}

bool SearchSession::fail(AliceSearch::EvalFailure& failure, AliceSearch::EvalFailureCode code,
                         AliceSearch::EvalStage stage, int perspective) const noexcept {
    failure             = {};
    failure.code        = code;
    failure.stage       = stage;
    failure.generation  = parameterGeneration;
    failure.ply         = int(currentPly);
    failure.perspective = perspective;
    return false;
}

bool SearchSession::inference_fail(AliceSearch::EvalFailure& failure, std::string_view message,
                                   AliceSearch::EvalStage defaultStage,
                                   int perspective) const noexcept {
    AliceSearch::EvalFailureCode code  = AliceSearch::EvalFailureCode::INTERNAL_INVARIANT;
    AliceSearch::EvalStage       stage = defaultStage;
    if (message.find("feature index") != std::string_view::npos)
        code = AliceSearch::EvalFailureCode::FEATURE_INDEX_OUT_OF_RANGE;
    else if (message.find("PSQT") != std::string_view::npos)
        code = AliceSearch::EvalFailureCode::PSQT_OUT_OF_RANGE;
    else if (message.find("accumulator") != std::string_view::npos
             || message.find("feature refresh") != std::string_view::npos)
        code = AliceSearch::EvalFailureCode::ACCUMULATOR_OUT_OF_RANGE;
    else if (message.find("fc0") != std::string_view::npos)
    {
        code = AliceSearch::EvalFailureCode::DENSE_ARITHMETIC_OUT_OF_RANGE;
        stage = AliceSearch::EvalStage::FC0;
    }
    else if (message.find("fc1") != std::string_view::npos)
    {
        code = AliceSearch::EvalFailureCode::DENSE_ARITHMETIC_OUT_OF_RANGE;
        stage = AliceSearch::EvalStage::FC1;
    }
    else if (message.find("fc2") != std::string_view::npos)
    {
        code = AliceSearch::EvalFailureCode::DENSE_ARITHMETIC_OUT_OF_RANGE;
        stage = AliceSearch::EvalStage::FC2;
    }
    else if (message.find("value") != std::string_view::npos)
    {
        code = AliceSearch::EvalFailureCode::DENSE_ARITHMETIC_OUT_OF_RANGE;
        stage = AliceSearch::EvalStage::OUTPUT_SCALING;
    }
    return fail(failure, code, stage, perspective);
}

bool SearchSession::initialize(const Position& root) noexcept {
    if (!complete(parameters) || parameterGeneration == 0 || parameterSha256.empty())
        return fail(initializationFailure, AliceSearch::EvalFailureCode::NOT_READY,
                    AliceSearch::EvalStage::ROOT_REFRESH);
    Frame& rootFrame = frames[0];
    rootFrame.position = capture(root);
    AliceNative::PieceSnapshot rootSnapshot;
    if (auto error = AliceNative::build_piece_snapshot(root, rootSnapshot))
    {
        const auto code = error->find("capacity") != std::string::npos
                          ? AliceSearch::EvalFailureCode::FEATURE_CAPACITY_EXCEEDED
                          : AliceSearch::EvalFailureCode::UNSUPPORTED_POSITION;
        return fail(initializationFailure, code, AliceSearch::EvalStage::FEATURE_EXTRACTION);
    }
    for (Color perspective : {WHITE, BLACK})
    {
        rootFrame.kingSquares[perspective] = rootSnapshot[perspective].kingSquare;
        rootFrame.kingBoards[perspective]  = rootSnapshot[perspective].kingBoard;
        if (auto error = refresh_accumulator(parameters, rootSnapshot[perspective],
                                             rootFrame.accumulators[perspective]))
            return inference_fail(initializationFailure, *error,
                                  AliceSearch::EvalStage::ROOT_REFRESH, int(perspective));
    }
    return true;
}

bool SearchSession::evaluate(const Position& position, Value& value,
                             AliceSearch::EvalFailure& failure) noexcept {
    if (!initialized)
    {
        failure = initializationFailure;
        return false;
    }
    if (!matches_current(position))
        return fail(failure, AliceSearch::EvalFailureCode::POSITION_MISMATCH,
                    AliceSearch::EvalStage::EVALUATE);
    i32 result = 0;
    if (auto error = evaluate_value(parameters, position, frames[currentPly].accumulators,
                                    true, result))
        return inference_fail(failure, *error, AliceSearch::EvalStage::EVALUATE);
    if (result <= -VALUE_TB_WIN_IN_MAX_PLY || result >= VALUE_TB_WIN_IN_MAX_PLY)
    {
        fail(failure, AliceSearch::EvalFailureCode::STATIC_VALUE_OUT_OF_RANGE,
             AliceSearch::EvalStage::OUTPUT_SCALING);
        failure.observed = result;
        failure.minimum  = -VALUE_TB_WIN_IN_MAX_PLY + 1;
        failure.maximum  = VALUE_TB_WIN_IN_MAX_PLY - 1;
        return false;
    }
    value = Value(result);
    ++counters.evaluations;
    return true;
}

bool SearchSession::push(const Position& position, const Dirties& dirties,
                         AliceSearch::EvalFailure& failure) noexcept {
    if (!initialized)
    {
        failure = initializationFailure;
        return false;
    }
    if (currentPly >= MAX_PLY)
        return fail(failure, AliceSearch::EvalFailureCode::STACK_OVERFLOW,
                    AliceSearch::EvalStage::PUSH);
    const Frame& parent = frames[currentPly];
    if (position.state()->previous != parent.position.state)
        return fail(failure, AliceSearch::EvalFailureCode::POSITION_MISMATCH,
                    AliceSearch::EvalStage::PUSH);
    Frame& child = frames[currentPly + 1];
    child.position = capture(position);
    child.accumulators = parent.accumulators;
    child.nullTransition = false;
    for (Color perspective : {WHITE, BLACK})
    {
        child.kingSquares[perspective] = position.square<KING>(perspective);
        if (child.kingSquares[perspective] == SQ_NONE)
            return fail(failure, AliceSearch::EvalFailureCode::UNSUPPORTED_POSITION,
                        AliceSearch::EvalStage::FEATURE_EXTRACTION, int(perspective));
        child.kingBoards[perspective] = position.board_of(child.kingSquares[perspective]);
    }

    AliceNative::PieceSnapshot refreshSnapshot;
    bool                       refreshSnapshotReady = false;
    for (Color perspective : {WHITE, BLACK})
    {
        if (parent.kingSquares[perspective] != child.kingSquares[perspective]
            || parent.kingBoards[perspective] != child.kingBoards[perspective])
        {
            if (!refreshSnapshotReady)
            {
                if (auto error = AliceNative::build_piece_snapshot(position, refreshSnapshot))
                {
                    const auto code = error->find("capacity") != std::string::npos
                                      ? AliceSearch::EvalFailureCode::FEATURE_CAPACITY_EXCEEDED
                                      : AliceSearch::EvalFailureCode::UNSUPPORTED_POSITION;
                    return fail(failure, code, AliceSearch::EvalStage::FEATURE_EXTRACTION,
                                int(perspective));
                }
                refreshSnapshotReady = true;
            }
            if (auto error = refresh_accumulator(parameters, refreshSnapshot[perspective],
                                                 child.accumulators[perspective]))
                return inference_fail(failure, *error, AliceSearch::EvalStage::ROOT_REFRESH,
                                      int(perspective));
            ++counters.fullRefreshes[perspective];
        }
        else
        {
            const Square kingSquare = parent.kingSquares[perspective];
            const Board  kingBoard  = parent.kingBoards[perspective];
            PieceEventIndices removals{};
            PieceEventIndices additions{};
            usize removalCount = 0, additionCount = 0;
            const auto append = [&](PieceEventIndices& events, usize& count, Piece piece,
                                    Square square, Bitboard boardB) {
                if (count == events.size() || piece == NO_PIECE || !is_ok(square))
                    return false;
                events[count++] = AliceNative::PieceSquareFeatures::make_index(
                  perspective, square, piece, board_at(boardB, square), kingSquare, kingBoard);
                return true;
            };
            const DirtyPiece& dirty = dirties.dirtyPiece;
            if (!append(removals, removalCount, dirty.pc, dirty.from, parent.position.boardB)
                || (dirty.remove_sq != SQ_NONE
                    && !append(removals, removalCount, dirty.remove_pc, dirty.remove_sq,
                               parent.position.boardB))
                || (dirty.to != SQ_NONE
                    && !append(additions, additionCount, dirty.pc, dirty.to,
                               child.position.boardB))
                || (dirty.add_sq != SQ_NONE
                    && !append(additions, additionCount, dirty.add_pc, dirty.add_sq,
                               child.position.boardB)))
                return fail(failure, AliceSearch::EvalFailureCode::FEATURE_CAPACITY_EXCEEDED,
                            AliceSearch::EvalStage::FEATURE_EXTRACTION, int(perspective));

            AccumulatorDeltaStats delta;
            if (auto error = update_accumulator_events(
                  parameters, removals, removalCount, additions, additionCount,
                  child.accumulators[perspective], delta))
                return inference_fail(failure, *error, AliceSearch::EvalStage::PUSH,
                                      int(perspective));
            counters.pieceAdds += delta.pieceAdds;
            counters.pieceRemoves += delta.pieceRemoves;
            counters.maxPieceEvents =
              std::max(counters.maxPieceEvents, delta.pieceAdds + delta.pieceRemoves);
        }
    }
    ++currentPly;
    ++counters.pushes;
    return true;
}

bool SearchSession::push_null(const Position& position,
                              AliceSearch::EvalFailure& failure) noexcept {
    if (!initialized)
    {
        failure = initializationFailure;
        return false;
    }
    if (currentPly >= MAX_PLY)
        return fail(failure, AliceSearch::EvalFailureCode::STACK_OVERFLOW,
                    AliceSearch::EvalStage::PUSH);

    const Frame& parent = frames[currentPly];
    if (position.state()->previous != parent.position.state
        || position.state()->boardB != parent.position.boardB
        || position.count<ALL_PIECES>() != parent.position.pieceCount)
        return fail(failure, AliceSearch::EvalFailureCode::POSITION_MISMATCH,
                    AliceSearch::EvalStage::PUSH);

    Frame& child = frames[currentPly + 1];
    child = parent;
    child.position = capture(position);
    child.nullTransition = true;
    ++currentPly;
    ++counters.pushes;
    ++counters.nullPushes;
    return true;
}

bool SearchSession::pop(const Position& restoredParent,
                        AliceSearch::EvalFailure& failure) noexcept {
    if (!initialized)
    {
        failure = initializationFailure;
        return false;
    }
    if (currentPly == 0)
        return fail(failure, AliceSearch::EvalFailureCode::STACK_UNDERFLOW,
                    AliceSearch::EvalStage::POP);
    if (!same_position(frames[currentPly - 1].position, restoredParent))
        return fail(failure, AliceSearch::EvalFailureCode::POSITION_MISMATCH,
                    AliceSearch::EvalStage::POP);
    const bool nullTransition = frames[currentPly].nullTransition;
    --currentPly;
    ++counters.pops;
    counters.nullPops += nullTransition;
    return true;
}

bool SearchSession::ready() const noexcept { return initialized; }
bool SearchSession::matches_current(const Position& position) const noexcept {
    return initialized && same_position(frames[currentPly].position, position);
}
usize SearchSession::ply() const noexcept { return currentPly; }
const IntegerAccumulatorSet& SearchSession::current_accumulators() const noexcept {
    return frames[currentPly].accumulators;
}
const RuntimeSessionStats& SearchSession::stats() const noexcept { return counters; }

}  // namespace Stockfish::Eval::NNUE::AliceNativeV2
