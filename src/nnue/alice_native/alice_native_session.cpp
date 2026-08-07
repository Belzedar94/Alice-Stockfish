/*
  Alice-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "alice_native_session.h"

#include <algorithm>

#include "../../position.h"

namespace Stockfish::Eval::NNUE::AliceNative {

SearchSession::SearchSession(const ParameterView& loadedParameters,
                             u64                  generation,
                             std::string_view     sha256,
                             const Position&      root) noexcept :
    parameters(loadedParameters),
    parameterGeneration(generation),
    parameterSha256(sha256) {
    initialized = initialize(root);
}

AliceSearch::EvaluatorIdentity SearchSession::identity() const noexcept {
    return {"AliceNativeV1", parameterGeneration, parameterSha256};
}

SearchSession::PositionIdentity SearchSession::capture(const Position& position) noexcept {
    return {position.state(), position.key(), position.state()->boardB, position.side_to_move(),
            position.count<ALL_PIECES>()};
}

bool SearchSession::same_position(const PositionIdentity& expected,
                                  const Position&         position) noexcept {
    return expected.state == position.state() && expected.key == position.key()
        && expected.boardB == position.state()->boardB
        && expected.sideToMove == position.side_to_move()
        && expected.pieceCount == position.count<ALL_PIECES>();
}

bool SearchSession::complete(const ParameterView& view) noexcept {
    if (!view.ftBias || !view.threatWeight || !view.threatPsqt || !view.pieceSquareWeight
        || !view.pieceSquarePsqt)
        return false;
    for (const auto& dense : view.dense)
        if (!dense.fc0Bias || !dense.fc0Weight || !dense.fc1Bias || !dense.fc1Weight
            || !dense.fc2Bias || !dense.fc2Weight)
            return false;
    return true;
}

bool SearchSession::fail(AliceSearch::EvalFailure&    failure,
                         AliceSearch::EvalFailureCode code,
                         AliceSearch::EvalStage       stage,
                         int                          perspective) const noexcept {
    failure             = {};
    failure.code        = code;
    failure.stage       = stage;
    failure.generation  = parameterGeneration;
    failure.ply         = int(currentPly);
    failure.perspective = perspective;
    return false;
}

bool SearchSession::inference_fail(AliceSearch::EvalFailure& failure,
                                   std::string_view          message,
                                   AliceSearch::EvalStage    defaultStage,
                                   int                       perspective) const noexcept {
    AliceSearch::EvalFailureCode code  = AliceSearch::EvalFailureCode::INTERNAL_INVARIANT;
    AliceSearch::EvalStage       stage = defaultStage;

    if (message.find("feature index") != std::string_view::npos)
        code = AliceSearch::EvalFailureCode::FEATURE_INDEX_OUT_OF_RANGE;
    else if (message.find("PSQT accumulator") != std::string_view::npos)
        code = AliceSearch::EvalFailureCode::PSQT_OUT_OF_RANGE;
    else if (message.find("feature accumulator") != std::string_view::npos)
        code = AliceSearch::EvalFailureCode::ACCUMULATOR_OUT_OF_RANGE;
    else if (message.find("fc0") != std::string_view::npos)
    {
        code  = AliceSearch::EvalFailureCode::DENSE_ARITHMETIC_OUT_OF_RANGE;
        stage = AliceSearch::EvalStage::FC0;
    }
    else if (message.find("fc1") != std::string_view::npos)
    {
        code  = AliceSearch::EvalFailureCode::DENSE_ARITHMETIC_OUT_OF_RANGE;
        stage = AliceSearch::EvalStage::FC1;
    }
    else if (message.find("fc2") != std::string_view::npos)
    {
        code  = AliceSearch::EvalFailureCode::DENSE_ARITHMETIC_OUT_OF_RANGE;
        stage = AliceSearch::EvalStage::FC2;
    }
    else if (message.find("fwdOut") != std::string_view::npos
             || message.find("positional") != std::string_view::npos
             || message.find("psqtRaw") != std::string_view::npos
             || message.find("native value") != std::string_view::npos)
    {
        code  = AliceSearch::EvalFailureCode::DENSE_ARITHMETIC_OUT_OF_RANGE;
        stage = AliceSearch::EvalStage::OUTPUT_SCALING;
    }
    return fail(failure, code, stage, perspective);
}

bool SearchSession::initialize(const Position& root) noexcept {
    if (!complete(parameters) || parameterGeneration == 0 || parameterSha256.empty())
        return fail(initializationFailure, AliceSearch::EvalFailureCode::NOT_READY,
                    AliceSearch::EvalStage::ROOT_REFRESH);

    Frame& rootFrame   = frames[0];
    rootFrame.position = capture(root);
    if (auto error = build_fixed_snapshot(root, rootFrame.snapshot))
    {
        const auto code = error->find("capacity") != std::string::npos
                          ? AliceSearch::EvalFailureCode::FEATURE_CAPACITY_EXCEEDED
                          : AliceSearch::EvalFailureCode::UNSUPPORTED_POSITION;
        return fail(initializationFailure, code, AliceSearch::EvalStage::FEATURE_EXTRACTION);
    }

    for (Color perspective : {WHITE, BLACK})
        if (auto error = refresh_integer_accumulator(parameters, rootFrame.snapshot[perspective],
                                                     rootFrame.accumulators[perspective]))
            return inference_fail(initializationFailure, *error,
                                  AliceSearch::EvalStage::ROOT_REFRESH, int(perspective));

    return true;
}

bool SearchSession::evaluate(const Position&           position,
                             Value&                    value,
                             AliceSearch::EvalFailure& failure) noexcept {
    if (!initialized)
    {
        failure = initializationFailure;
        return false;
    }
    if (!matches_current(position))
        return fail(failure, AliceSearch::EvalFailureCode::POSITION_MISMATCH,
                    AliceSearch::EvalStage::EVALUATE);

    NativeIntegerStages stages;
    if (auto error =
          evaluate_integer(parameters, position, frames[currentPly].accumulators, stages))
        return inference_fail(failure, *error, AliceSearch::EvalStage::EVALUATE);
    if (stages.value <= -VALUE_TB_WIN_IN_MAX_PLY || stages.value >= VALUE_TB_WIN_IN_MAX_PLY)
    {
        fail(failure, AliceSearch::EvalFailureCode::STATIC_VALUE_OUT_OF_RANGE,
             AliceSearch::EvalStage::OUTPUT_SCALING);
        failure.observed = stages.value;
        failure.minimum  = -VALUE_TB_WIN_IN_MAX_PLY + 1;
        failure.maximum  = VALUE_TB_WIN_IN_MAX_PLY - 1;
        return false;
    }

    value = stages.value;
    ++counters.evaluations;
    return true;
}

bool SearchSession::push(const Position& position,
                         const Dirties&,
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

    Frame& child       = frames[currentPly + 1];
    child.position     = capture(position);
    child.accumulators = parent.accumulators;
    if (auto error = build_fixed_snapshot(position, child.snapshot))
    {
        const auto code = error->find("capacity") != std::string::npos
                          ? AliceSearch::EvalFailureCode::FEATURE_CAPACITY_EXCEEDED
                          : AliceSearch::EvalFailureCode::UNSUPPORTED_POSITION;
        return fail(failure, code, AliceSearch::EvalStage::FEATURE_EXTRACTION);
    }

    for (Color perspective : {WHITE, BLACK})
    {
        const auto& before = parent.snapshot[perspective];
        const auto& after  = child.snapshot[perspective];
        if (before.kingSquare != after.kingSquare || before.kingBoard != after.kingBoard)
        {
            if (auto error =
                  refresh_integer_accumulator(parameters, after, child.accumulators[perspective]))
                return inference_fail(failure, *error, AliceSearch::EvalStage::ROOT_REFRESH,
                                      int(perspective));
            ++counters.fullRefreshes[perspective];
        }
        else
        {
            AccumulatorDeltaStats delta;
            if (auto error = update_integer_accumulator(parameters, before, after,
                                                        child.accumulators[perspective], delta))
                return inference_fail(failure, *error, AliceSearch::EvalStage::PUSH,
                                      int(perspective));
            counters.pieceAdds += delta.pieceAdds;
            counters.pieceRemoves += delta.pieceRemoves;
            counters.threatAdds += delta.threatAdds;
            counters.threatRemoves += delta.threatRemoves;
            counters.maxPieceEvents =
              std::max(counters.maxPieceEvents, delta.pieceAdds + delta.pieceRemoves);
            counters.maxThreatEvents =
              std::max(counters.maxThreatEvents, delta.threatAdds + delta.threatRemoves);
        }
    }

    ++currentPly;
    ++counters.pushes;
    return true;
}

bool SearchSession::pop(const Position&           restoredParent,
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

    --currentPly;
    ++counters.pops;
    return true;
}

bool SearchSession::ready() const noexcept { return initialized; }

usize SearchSession::ply() const noexcept { return currentPly; }

bool SearchSession::matches_current(const Position& position) const noexcept {
    return initialized && same_position(frames[currentPly].position, position);
}

const FeatureSnapshot& SearchSession::current_snapshot() const noexcept {
    return frames[currentPly].snapshot;
}

const IntegerAccumulatorSet& SearchSession::current_accumulators() const noexcept {
    return frames[currentPly].accumulators;
}

const RuntimeSessionStats& SearchSession::stats() const noexcept { return counters; }

}  // namespace Stockfish::Eval::NNUE::AliceNative
